# FutureSeed / GDN2 / Momentum 论文思路伪代码

这份文档只解释已经获得实验证据的主线。它刻意把三类东西分开：

1. **可训练、可部署的方法**：FutureSeed、重复深度循环、position-Q/K、Momentum recurrence、train-full/serve-Momentum。
2. **用于理解失败原因的诊断**：错键正确值分类、真实提交残差、因果关写实验。
3. **不能声称已经完成的工作**：Momentum 基础算子不是本项目发明；Momentum 结果尚未转移到 hard Sudoku；诊断中使用标签的操作不能进入模型。

## 0. 先认识所有符号

为了把公式讲清楚，下面先忽略 batch 和多头维度，只写单个样例、单个 head。

```text
T            序列长度
K            key/address 维度
V            value/payload 维度
x[t]         第 t 个 token 的隐藏向量
q[t]         读取地址，shape [K]
k[t]         写入地址，shape [K]
v[t]         候选写入内容，shape [V]
g[t]         旧记忆的逐地址衰减，shape [K]
b[t]         擦除门，shape [K]
w[t]         写入门，shape [V]
S            GDN2 记忆矩阵，shape [K, V]
outer(k, e)  外积，结果 shape [K, V]
RMS(A)       sqrt(mean(A^2))
```

可以把 `S` 想成一张可微分的键值表。`q^T @ S` 是按地址 `q` 读表，`outer(k,e)` 是把内容 `e` 写到地址 `k`。

## 1. 基础：GDN2 到底怎样读写记忆

下面是 official GDN2 recurrence 的概念等价写法。实际训练使用 chunk/Triton kernel，并不会真的写 Python token 循环。

```text
function GDN2_LAYER(X, initial_state=None):
    Q = q_projection_and_short_conv(X)
    K = k_projection_and_short_conv(X)
    V = v_projection_and_short_conv(X)
    G = negative_decay_projection(X)
    B = sigmoid(erase_projection(X))
    W = sigmoid(write_projection(X))

    S = zeros([K_dim, V_dim]) if initial_state is None else initial_state

    for t in 0 .. T-1:
        q = normalize(Q[t])
        k = normalize(K[t])

        # 第一步：衰减旧记忆。不同 key 方向可以有不同保留率。
        S_decayed = diag(exp(G[t])) @ S

        # 第二步：旧记忆在擦除地址上已经存了什么。
        old_value = (B[t] * k)^T @ S_decayed

        # 第三步：当前 token 真正还需要提交多少新内容。
        desired_value = W[t] * V[t]
        committed_residual = desired_value - old_value

        # 第四步：把这个残差写回当前 key。
        S = S_decayed + outer(k, committed_residual)

        # 第五步：用 query 读更新后的状态。
        read[t] = q^T @ S / sqrt(K_dim)

    Y = output_projection(output_norm(read, X))
    return Y, S
```

最关键的量是：

```text
e[t] = committed_residual
     = W[t] * V[t] - (B[t] * k[t])^T @ S_decayed
```

`e[t]` 不是一个额外学习出来的分数，而是 GDN2 这一步真正写进记忆的内容。后面的“surprise”诊断就是 `norm(e[t])`。

## 2. 方法一：FutureSeed

普通 causal stack 每层都从零状态开始：

```text
layer 0: scan token 0 -> token T-1, initial state = 0
layer 1: scan token 0 -> token T-1, initial state = 0
```

FutureSeed 把上一层的最终状态，变成下一层的初始状态：

```text
function NORMALIZE_AND_GATE(terminal_state, gate_logit, scale):
    # 对每个样例、每个 head 独立归一化整个 KxV 矩阵。
    normalized = terminal_state / max(RMS(terminal_state), 1e-6)
    gate = sigmoid(gate_logit)       # 每个接收层、每个 head 一个可学习门
    return scale * gate * normalized


function FUTURESEED_STACK(H):
    previous_terminal = None

    for layer in 0 .. num_layers-1:
        if previous_terminal is None:
            seed = None
        else:
            seed = NORMALIZE_AND_GATE(
                previous_terminal,
                gate_logit[layer],
                future_seed_scale,
            )

        H = residual_norm_before_mixer(H)
        mixer_output, terminal = GDN2_LAYER(H, initial_state=seed)
        H = residual_and_channel_mixer(H, mixer_output)
        previous_terminal = terminal

    return final_norm(H)
```

为什么它带来了未来信息？假设 token 5 询问一个答案，而相关写入在 token 900：

```text
第一层处理 token 5 时：还没见过 token 900，因此不知道答案。
第一层处理到 token 1023 后：terminal state 已经包含 token 900 的写入。
第二层处理 token 5 前：先拿到第一层 terminal state 作为 initial state。
第二层处理 token 5 时：可以从这个状态读到 token 900 的信息。
```

因此 FutureSeed 不是右到左扫描。每层仍然严格左到右，只是深层从浅层已经完成的全序列摘要开始。

## 3. 方法二：重复深度循环怎样纠错

论文级最简版本可以写成：

```text
function REPEATED_REASONING(input_tokens, loops):
    X = token_embedding(input_tokens) + position_embedding
    H = learned_zero_state_like(X)

    for r in 0 .. loops-1:
        proposal = FUTURESEED_STACK(H + X)
        H = H + update_rate * (proposal - H)
        logits[r] = classifier(normalize(H))

    loss = mean(cross_entropy(logits[r], labels) for every r)
    return logits
```

实际 Sudoku runner 有一个低层工作流 `Z_low` 和一个高层工作流 `Z_high`：

```text
function DOUBLE_LOOP_SUDOKU(board, loops, low_cycles):
    X = embed(board) + canonical_position_embedding
    Z_low  = learned_initial_low
    Z_high = learned_initial_high

    for r in 0 .. loops-1:
        for c in 0 .. low_cycles-1:
            P_low = FUTURESEED_STACK(Z_low + Z_high + X)
            Z_low = blend(Z_low, P_low)

        P_high = FUTURESEED_STACK(Z_high + Z_low)
        Z_high = blend(Z_high, P_high)

        logits[r] = sudoku_head(normalize(Z_high))

    loss = mean(CE(logits[r], solution) for every r)
    return logits[0 .. loops-1]
```

这里没有 Sudoku 规则、搜索或 repair。每一轮只是让同一个可学习 reasoner 再处理一次当前隐藏状态；对每轮都加监督，防止只有最后一轮会工作。

## 4. 方法三：地址和内容分工的 position-Q/K GDN2

普通 GDN2 所有投影都读同一个内容隐藏向量：

```text
q,k,v,g,b,w = projections(x[t])
```

position-Q/K 只改变 Q/K 的输入：

```text
function POSITION_QK_GDN2(X, canonical_positions, traversal=None, seed=None):
    A = position_embedding(canonical_positions)

    # 地址只由稳定的 canonical cell position 产生。
    Q_canonical = q_short_conv(q_projection(A))
    K_canonical = k_short_conv(k_projection(A))

    # 内容和所有修改强度仍由当前隐藏内容产生。
    V = v_short_conv(v_projection(X))
    G = decay_projection(X)
    B = sigmoid(erase_projection(X))
    W = sigmoid(write_projection(X))
    O_gate = output_gate_projection(X)

    if traversal is not None:
        # 例如随机访问数独格子时，Q/K 跟随格子的 canonical 身份移动。
        Q = gather(Q_canonical, traversal)
        K = gather(K_canonical, traversal)
    else:
        Q, K = Q_canonical, K_canonical

    read, terminal = OFFICIAL_GDN2_SCAN(Q, K, V, G, B, W, seed)
    return output_projection(output_norm(read, O_gate)), terminal
```

一句话：**位置决定写到哪里，内容决定写什么、擦多少、输出多少。**

它在 Sudoku 有意义，因为每个格子有稳定的 canonical 身份。它不能直接推广成“所有任务都应把 key 从内容中拿掉”；MQAR 的 owner 是动态键值事件，不是固定格子位置。

## 5. 方法四：Momentum 二阶循环记忆

普通 GDN2 只有当前状态 `S`。Momentum recurrence 再维护一个更新趋势 `M`。P059 使用的核心概念伪代码是：

```text
function MOMENTUM_LAYER(X, initial_pair=None):
    q,k,v,alpha,mu,beta,eta,read_correction = projections(X)

    if initial_pair is None:
        S = zeros([K_dim, V_dim])
        M = zeros([K_dim, V_dim])
    else:
        S, M = initial_pair

    for t in 0 .. T-1:
        # 当前 S 在 owner key 上预测了什么；P059 中 prediction key p = k。
        predicted = S^T @ (alpha[t] * k[t])
        value_residual = v[t] - predicted

        # M 累积连续的更新方向。mu 控制旧趋势保留多少。
        M = mu[t] * M - outer(k[t], eta[t] * value_residual)

        # S 不再直接吃当前残差，而是由已经积累的 M 推动。
        S = alpha[t] * S - beta[t] * M

        # 外部 Momentum DeltaNet 还带一个必要的学习式 read correction。
        q_read = q[t] - exp(read_correction[t]) * k[t]
        read[t] = S^T @ q_read

    return output_projection(read), stack([S, M])
```

直觉：

```text
一次偶然的错误写入        -> 只短暂影响 M，未必立刻重写全部 S
连续一致的绑定更新        -> 在 M 中同向累积，再稳定推动 S
```

这是“改变活记忆的更新规律”，不是在旧 GDN2 外面加一个读取器。Momentum DeltaNet 基础算子来自外部项目；本项目的贡献是严格接入、FutureSeed 组合、机制分析和部署压缩，不能声称发明了基础 recurrence。

## 6. 方法五：让 FutureSeed 传 `[S,M]`

Momentum layer 返回两个矩阵。训练时的 native Momentum FutureSeed 是：

```text
function MOMENTUM_FUTURESEED(terminal_pair, gate_logit):
    S_terminal, M_terminal = terminal_pair

    # 两个分量分别归一化，不能用一个大 RMS 混在一起。
    S_seed = S_terminal / max(RMS(S_terminal), 1e-6)
    M_seed = M_terminal / max(RMS(M_terminal), 1e-6)
    gate = sigmoid(gate_logit)

    return stack([
        future_seed_scale * gate * S_seed,
        future_seed_scale * gate * M_seed,
    ])
```

然后下一层调用：

```text
Y, terminal_pair = MOMENTUM_LAYER(X, initial_pair=seed_pair)
```

第一层完整扫过序列后，不只把“最终记忆”交给下一层，还把“记忆是怎样变化到这里的趋势”交给下一层。

## 7. 方法六：Train-Full / Serve-Momentum

这是训练和部署不对称的方法。

训练阶段必须使用完整 `[S,M]`：

```text
function TRAIN_FORWARD(X):
    terminal_pair = None
    for layer in layers:
        seed_pair = None if terminal_pair is None else MOMENTUM_FUTURESEED(
            terminal_pair,
            layer.gate_logit,
        )
        X, terminal_pair = MOMENTUM_LAYER(X, seed_pair)
    return logits(X)
```

训练完成后冻结所有参数。部署阶段只发送 `M`：

```text
function DEPLOY_FORWARD_M_ONLY(X):
    momentum_payload = None

    for layer in layers:
        if momentum_payload is None:
            seed_pair = None
        else:
            M_seed = momentum_payload / max(RMS(momentum_payload), 1e-6)
            M_seed = sigmoid(layer.gate_logit) * M_seed
            seed_pair = stack([zeros_like(M_seed), M_seed])

        X, terminal_pair = MOMENTUM_LAYER(X, seed_pair)
        S_terminal, M_terminal = terminal_pair

        # 跨层只打包 M；S 不进入传输 payload。
        momentum_payload = M_terminal

    return logits(X)
```

注意这里没有微调、蒸馏或新参数。它只改变部署时跨层传输的数据：

```text
native payload: [S, M] = 8192 values
M-only payload: [M]    = 4096 values
receiver rebuilds:     [0, normalized(M)]
```

为什么不能从训练第一步就这样做？因为实验表明 `S` 虽然在最终推理时几乎可以不传，但训练过程中它帮助上下层共同形成可用的 `M`。这叫“训练脚手架”，不是“无用状态”。

## 8. 诊断一：错键正确值怎样分类

下面只用于评估，不允许进入模型：

```text
function CLASSIFY_MQAR_ERROR(example, query, prediction):
    target_value = example.answer_for(query.target_key)

    # 把这个序列里出现过的每个 value 映射回它的 owner key。
    value_to_owner = {}
    for (key, value) in example.associations:
        value_to_owner[value] = key

    if prediction == target_value:
        return "correct"

    if prediction in value_to_owner:
        wrong_owner = value_to_owner[prediction]
        return "wrong-key valid-value", wrong_owner

    return "other wrong value"
```

若模型输出的是另一个关联中存在的正确 value，就说明内容还在记忆里，但 owner 绑定错了。不能只看“wrong-key 占剩余错误的比例”，还必须同时报告总错误数；否则一个完全崩溃的模型也可能因为随机输出而显得 wrong-key 比例很低。

## 9. 诊断二：真实提交残差和 surprise

不改 logits，只记录 GDN2 本来就在算的量：

```text
function RECORD_COMMITTED_EDIT_AT_TOKEN(t, S_previous):
    S_decayed = diag(exp(g[t])) @ S_previous
    old_value = (b[t] * k[t])^T @ S_decayed
    desired_value = w[t] * v[t]
    e = desired_value - old_value

    surprise = l2_norm(e)
    state_edit = outer(k[t], e)

    return {
        "committed_residual": e,
        "surprise": surprise,
        "state_edit_frobenius": frobenius_norm(state_edit),
    }
```

如果 `k` 已归一化，那么 `norm(e)` 与 `norm(outer(k,e))` 等价。它回答的是：“这个 token 实际给记忆带来了多大的新信息？”而不是让另一个网络猜重要性。

## 10. 诊断三：因果关写实验

这个诊断使用真实答案定位 owner，只能用于论文分析，绝不能作为推理算法：

```text
function CAUSAL_WRONG_OWNER_TEST(model, swap_case):
    target_owner = swap_case.correct_key
    predicted_owner = owner_of(swap_case.predicted_valid_value)

    baseline_prediction = model(swap_case.input)

    # 只在指定层、指定 write token，把竞争 owner 的写入门设为零。
    intervention = {
        "layer": 0,
        "token": write_token_of(predicted_owner),
        "force_write_gate_to_zero": True,
    }
    changed_prediction = model(swap_case.input, intervention)

    repaired = changed_prediction == swap_case.correct_value
    return repaired
```

当关掉错误 owner 的一次写入能修复大量样例时，说明失败发生在多个 owner 写入同一活记忆时的相互覆盖，而不是输出分类头不知道哪个数字合法。

## 11. 这些方法怎样连成一条论文逻辑

```text
普通 causal recurrence
    |
    | 未来 token 不可见
    v
FutureSeed：把浅层全序列 terminal state 给深层
    |
    | L64-L512 几乎解决未来查询；L1024 出现 cliff
    v
错误诊断：值大多还在，但被相邻 key 读走
    |
    | 说明主要问题是 binding/ownership，不是 value capacity
    v
Position-Q/K：固定位置任务里，让地址和 payload 分工
    |
    | Sudoku blank/CE 明显改善，但整盘 closure 仍不足
    v
Momentum recurrence：直接改变 live state transition
    |
    | L1024 错误 2024 -> 223，swap 1546 -> 151
    v
Component diagnostic：跨层真正有用的主要是 M
    |
    v
Train full [S,M], serve M-only：传输量减半，质量基本不变
```

## 12. 哪些属于本项目的论文贡献

可以作为方法贡献：

- FutureSeed 的跨层 terminal-state initialization。
- FutureSeed 与重复深度循环的 reasoner 组合及跨 carrier 验证。
- Sudoku 的 position-Q/K address/payload factorization。
- Momentum recurrence 与 native `[S,M]` FutureSeed 的组合和严格机制验证。
- train-full/serve-Momentum 的 phase-asymmetric deployment contract。

可以作为机制发现：

- L1024 的主要错误是正确 value 归属到错误 key，而非 value 丢失。
- Q/K/erase/write/read 是共同训练的坐标系统，事后只“修好”一个部分通常会破坏整体。
- 更多状态容量不等于更好的可读取记忆。
- 推理时可删除的分量，不一定能在训练时删除。

不能声称：

- 本项目发明了 Momentum DeltaNet 基础 recurrence。
- MQAR 成功等于 hard Sudoku 成功。
- M-only 可以从零训练。
- 使用标签的 owner intervention 是部署算法。
- 当前实现已经比 BERT 更快或更省显存。

## 13. 对应实现位置

- Native GDN2 FutureSeed：`experiments/zoology_mqar/gdn2_futureseed.py`
- Momentum + FutureSeed：`experiments/zoology_mqar/momentum_futureseed.py`
- M-only deployment：`experiments/zoology_mqar/momentum_only_futureseed.py`
- Sudoku loops 与 position-Q/K：`experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py`
- Momentum 实验报告：`research/reports/experiments/gdn3-momentum-futureseed-mqar-20260816.md`
- Train-full/serve-Momentum 报告：`research/reports/experiments/futureseed2-train-full-serve-momentum-mqar-20260816.md`

