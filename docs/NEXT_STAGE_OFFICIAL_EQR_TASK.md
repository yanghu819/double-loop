# Next Stage Task: Official EqR FutureSeed Comparison

Date: 2026-06-22

## 1. Task Objective

在官方 EqR codebase 上，公平比较 EqR mixer baseline 和 FutureSeed 替代/压缩 mixer 的版本，判断 FutureSeed 是否能作为更便宜的双向信息机制，而不是只在 Sudoku/RWKV proxy 上有效。

核心论文问题：

- EqR mixer 是强 noncausal baseline。
- FutureSeed 的主张不是再给 EqR 加一个小 patch，而是让 recurrent/causal backbone 用更便宜的方式获得未来侧信息。
- 因此实验必须区分“完整 EqR mixer 已经做到的信息交换”和“FutureSeed 能否替代或压缩这部分信息交换”。

## 2. Success Criteria

必须跑出 matched-compute 对比：

- 同一个官方 EqR 数据生成/加载路径。
- 同一个训练预算。
- 同一个评估脚本。
- 同一个 path-aware 指标。
- 清楚记录参数量、训练速度、推理速度和显存。

每个对比至少报告：

- exact
- token accuracy
- path F1
- path precision
- path recall
- pred_path_frac
- FP / FN
- training wall time
- evaluation throughput
- peak or allocated VRAM
- parameter count
- representative hard-case visualizations

成功只按客观门槛判断：

- FutureSeed 版在 hard Maze 或更合适 proxy 上 path F1 提升 `>= +0.03`，且不是靠 pred_path_frac 膨胀、broad mask、FP 爆炸换来的；或
- FutureSeed 版达到同等 path F1，但训练或推理计算节省 `>= 20%`。

如果没赢，也必须输出：

- hard-case 可视化；
- loop1 到 final loop 的 FP/FN 变化；
- 失败归因：是信息机制不够、目标函数鼓励 broad mask、官方 proxy 不适合，还是 EqR mixer 已经覆盖了 FutureSeed 的价值。

## 3. Stop Lines

出现以下情况必须停，不继续烧资源：

- 官方 EqR baseline 不能复现。
- 数据、训练预算、评估脚本或指标不一致。
- 只靠 token accuracy 变好，但 path F1 / FP / FN 没有改善。
- FutureSeed 结果只是 pred_path_frac 变大、recall 变高，同时 FP 明显变多。
- 需要 selector、search、repair、oracle rollout 或 human rule 才能涨分。
- 连续两轮实验只是在扫 seed、loss weight、temperature、bias，而没有新的机制 insight。

## 4. References And Evidence Sources

优先查：

- 官方 EqR repository。
- EqR paper。
- 官方 EqR config / training / evaluation code。

本 repo 内优先查：

- `docs/PAPER_PLAN.md`
- `docs/LESSONS.md`
- `plans.md`
- `leaderboard.csv`
- `research/reports/experiments/`
- `runs/*/comparison.json`
- `runs/*/index.html`

卡住时顺序：

1. 先查官方 EqR 是否被正确复现。
2. 再查数据与 metric 是否一致。
3. 最后查我们已归档的 Maze/Sudoku 失败形态。

## 5. Boundaries

不能动：

- 不能改官方 EqR baseline 语义。
- 不能加 Sudoku/Maze repair。
- 不能加 search、selector、oracle rollout。
- 不能加任务特定路径规则。
- 不能用 GPU2。
- 不能做 CPU smoke。
- 不能提交模型、checkpoint、大数据或大 source snapshot 到 Git。
- 不能为了补表跑低 ROI ablation。
- 不能把完整 EqR mixer 和 FutureSeed 混成说不清贡献的模型，除非实验目的明确是对照。

必须保持：

- GPU1 only。
- 实验绑定 git SHA。
- 每个实验先写 hypothesis、prediction、budget、kill criteria。
- 每个实验归档 config、score、logs、visuals、comparison JSON。
- 结果必须能支持一个清楚判断：FutureSeed 是否能作为 cheap bidirectional information mechanism。

## First Concrete Gate

下一步不是继续调 Maze loss，而是做一个官方 EqR gate：

1. 复现官方 EqR Maze baseline，保持官方 baseline 语义不变。
2. 在同一 codebase 中实现一个贡献可分离的 FutureSeed 版本：
   - 要么替代一部分 mixer；
   - 要么压缩 mixer 容量后用 FutureSeed 补偿；
   - 不把完整 mixer 和 FutureSeed 简单叠加成无法归因的模型。
3. 只跑一条最高信息量 matched-compute probe。
4. 如果 baseline 复现失败，立即停下修复复现，不跑 FutureSeed。
5. 如果 FutureSeed 只扩大 broad mask，立即停下并可视化失败，不扫超参。

## 2026-06-22 Gate Result

Completed `official-eqr-causal-fs-gate-20260622` on GPU1.

- Base condition: official EqR code path with causalized attention.
- FutureSeed condition: same causalized path plus FutureSeed hooks.
- Dataset/objective: official `maze-30x30-unique-1k`, path-token weight `8`.
- Budget: e256 pair, 1792 estimated train steps, path-aware 256-case visual eval.

Result:

- causal base loop16 path F1: `0.468067`
- causal+FutureSeed loop16 path F1: `0.468157`
- delta: `+0.000090`
- precision/recall/pred PATH fraction stay effectively unchanged:
  about `0.306 / 1.0 / 0.433`
- false positives stay about `270` per case
- loop1 to loop16 gain remains tiny in both arms

Decision: this causalized-mixer gate is not positive evidence. Do not sweep
seed, gate bias, path weight, or causal attention details. The next official EqR
question is mixer compression: can FutureSeed recover performance when part of
the EqR mixer compute is removed? If not, Maze should remain a failure-analysis
proxy rather than a main positive benchmark.
