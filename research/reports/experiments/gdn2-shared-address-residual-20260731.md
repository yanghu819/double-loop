# GDN2 Shared Address Residual

## 1. Metainfo

- Plan: `P-ABIND-003`
- Status: in progress
- Approved by user: 2026-07-31
- Machine: AIStation GPU1, one A800 80GB
- Source branch: `codex/gdn2-address-operator-20260731`
- Parent checkpoint source: `42102bd65a28d60bde6b09ef93343692740582a1`

## 2. Hypothesis

Fixed and fully learned orthogonal address rotations both failed despite being
active. The learned field moved Q/K by roughly one vector norm but did not
create useful loop correction. Rotation preserves each content-vector norm
and expresses addresses indirectly as relative angles; that constraint is not
what the strong pure position-Q/K result supports.

Pure position-Q/K instead learns direct Euclidean directions for stable
addresses. A simpler generic integration is to add one shared, stable address
vector directly to both content Q and K. Shared coordinates make the same
address available to state writes and reads, while official in-kernel Q/K
normalization controls magnitude.

## 3. Mechanism And Configuration

For each layer and token:

```text
address = W_address * LN(stable token-plus-position anchor)
q_bound = q_content + address
k_bound = k_content + address
```

`W_address:D->D` has no bias and initializes to exact zero. Read and write use
the same residual by construction. It adds no recurrent state and does not
change V, decay, erase, write, output, rank-1 recurrence, FutureSeed, loops, or
the official FLA GDN2 kernel. At D192/L10 it adds 368,640 parameters (6.75%).

The anchor is ordinary token-plus-position input, fixed across loops. No row,
column, box, solver rule, search, repair, selector, or oracle is present. The
same operation transfers directly to language tokens.

Formal configuration matches the exact step9000 parent and all previous
step9100 gates: D192/L10/H6/K32, five loops, random traversal, all-loop CE,
official Sudoku data, batch32 x grad-accum4, seed52, and strict official FLA
SHA `9c8e42e7`. Only `GDN2_ADDRESS_MODE=anchor_residual` changes.

## 4. Environment Contract

- GPU1 only; exact UUID and `CUDA_VISIBLE_DEVICES=0` required.
- No GPU2 and no CPU model smoke.
- Fail closed on official FLA source/class/kernel/short-conv backend,
  checkpoint hash, source cleanliness, or GPU identity mismatch.
- All remote state remains below `/huyang2/double-loop`.

## 5. Prediction, Budget, And Kill Criteria

CUDA validation must prove exact zero-init official output/state/base-gradient
identity, nonzero first-step address-weight gradient, active input/anchor/state/
weight gradients, reorder consistency, and official chunk backward provenance.

Run one smoke, then one step9000->9100 formal candidate.

- Primary: mean official 51-64 loop5 blank accuracy at least `+0.10` above
  normal GDN2, CE at least `0.20` lower, genuine loop correction, runtime
  overhead at most 20%.
- Strong: mean hard blank at least `0.40`, approaching pure position-Q/K.
- Mechanism: address residual RMS and token variation nonzero, with content
  overwrite ratio finite and not explosively dominant.
- Kill at step9100 if mean delta is below `+0.05`, CE above `1.70`, address
  residual is inactive or simply overwrites content, or loops do not correct.
- Continue unchanged to step9300 only on primary pass.

One candidate only. No separate Q/K residual, scalar, rank, width, seed, LR,
loss, or initialization table. Budget: about 25 minutes on GPU1.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Decision And Lessons

Pending.

## 9. Submission

Not applicable.
