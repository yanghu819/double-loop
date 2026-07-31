# GDN2 Decoupled Read/Write Address Residual

## 1. Metainfo

- Plan: `P-ABIND-004`
- Status: in progress
- Approved by user: 2026-07-31
- Machine: AIStation GPU1, one A800 80GB
- Source branch: `codex/gdn2-address-operator-20260731`
- Parent checkpoint source: `42102bd65a28d60bde6b09ef93343692740582a1`

## 2. Hypothesis

The shared Euclidean address residual is the first generic mechanism here to
improve CE, all hard ranges, and later-loop correction. It still trails pure
position-Q/K. The sharp causal difference is symmetry: shared residual forces
read Q and write K to use the same address map, whereas normal attention and
the successful position-Q/K diagnostic use independent maps.

Independent read/write address maps can learn an asymmetric bilinear relation
between query identity and state-write coordinates. If that missing freedom is
the bottleneck, decoupling Q/K address projections should improve over the
shared residual without changing recurrent state or GDN2's update.

## 3. Mechanism And Configuration

```text
q_address = W_address_q * LN(stable token-plus-position anchor)
k_address = W_address_k * LN(stable token-plus-position anchor)
q_bound = q_content + q_address
k_bound = k_content + k_address
```

Both bias-free projections initialize to exact zero. Official in-kernel Q/K
normalization controls magnitude. V, decay, erase, write, output, rank-1
recurrence, recurrent state, FutureSeed, loops, and FLA kernel remain unchanged.
At D192/L10 this adds 737,280 parameters (13.5%), but no state.

This remains language-generic and contains no Sudoku structure. Formal setup
matches the exact step9000 parent, official data, random order, D192/L10/H6,
five loops, all-loop CE, batch32 x grad-accum4, seed52, and strict official FLA
SHA `9c8e42e7`. Only `GDN2_ADDRESS_MODE=anchor_qk_residual` changes.

## 4. Environment Contract

- GPU1 only; exact UUID and `CUDA_VISIBLE_DEVICES=0` required.
- No GPU2 and no CPU model smoke.
- Fail closed on official source/class/kernel/backend, checkpoint hash, source
  cleanliness, and GPU identity.
- All remote files remain below `/huyang2/double-loop`.

## 5. Prediction, Budget, And Kill Criteria

CUDA checks must prove exact zero-init official output/state/base-gradient
identity, nonzero first-step gradients for both maps, active input/anchor/state/
Q-map/K-map gradients, reorder consistency, and official chunk backward.

Run one smoke and one step9000->9100 formal gate.

- Primary: mean hard blank at least `+0.10` over normal, CE at least `0.20`
  lower, genuine loop correction, runtime overhead at most 25%.
- Mechanism improvement: beat shared residual mean `0.2844` by at least `+0.03`.
- Strong: mean hard blank at least `0.40`, approaching pure position-Q/K.
- Kill if mean is below `0.3144`, CE above `1.60`, loops fail to correct, either
  address map stays inactive, or address residuals explosively dominate content.
- Continue unchanged to step9300 only on primary plus mechanism-improvement pass.

No shared-vs-split table, rank, scalar, width, seed, LR, loss, or duration sweep.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Decision And Lessons

Pending.

## 9. Submission

Not applicable.
