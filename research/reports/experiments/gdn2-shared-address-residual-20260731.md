# GDN2 Shared Address Residual

## 1. Metainfo

- Plan: `P-ABIND-003`
- Status: done as a weak positive; no unchanged continuation authorized
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

- Formal run: `gdn2-address-anchor_residual-s9100-20260731T124800Z-7aa3f7b`
- Remote run directory:
  `/huyang2/double-loop/runs/gdn2-address-anchor_residual-s9100-20260731T124800Z-7aa3f7b`
- Local mirror:
  `.codex-transfer/localrepo/runs/gdn2-address-anchor_residual-s9100-20260731T124800Z-7aa3f7b`
- CUDA contract log:
  `/huyang2/double-loop/artifacts/launch/gdn2-anchor-residual-cuda-check-7aa3f7b.log`
- Exact clean source: `7aa3f7b182bd62a1dcfba40a1a9dd24e6ac1dcad`.
- Archive includes config, score, metadata, logs, source snapshot/SHA/patch,
  checkpoint eval, all case banks, JSON/Markdown, and HTML visualizations.

## 7. Results

CUDA validation passed against pinned official FLA `9c8e42e7`: zero-init
output/state/base-gradient errors were exactly `0.0` with and without initial
state; the new weight had a nonzero first-step gradient; active input, anchor,
state, and weight gradients were finite/nonzero; reorder error was `0.0`; and
the graph retained `ChunkGDN2FunctionBackward`.

| Step9100 metric | Normal GDN2 | Shared address residual | Delta |
|---|---:|---:|---:|
| train CE | 1.9370 | 1.6845 | -0.2525 better |
| 51-55 loop5 blank acc | 0.2120 | 0.3106 | +0.0986 |
| 56-60 loop5 blank acc | 0.2119 | 0.2860 | +0.0741 |
| 61-64 loop5 blank acc | 0.1871 | 0.2565 | +0.0694 |
| mean hard blank acc | 0.2037 | 0.2844 | +0.0807 |
| train elapsed | 649.9 s | 776.9 s | +19.5% |
| parameters | 5,461,688 | 5,830,328 | +368,640 |

The residual stayed materially smaller than content Q/K: at step9100 its
mean RMS was `0.5231`, with Q/K relative ratios `0.2782/0.3651`. Unlike either
rotation, it improved every official range from loop1 to loop5:
`0.3008->0.3106`, `0.2763->0.2860`, and `0.2492->0.2565`. Exact remains zero.

All paired data hashes match. On mechanically selected 64-blank batch 133,
wrong cells change normal `55->58`, pure position-Q/K `42->40`, and shared
residual `51->46->47->44->44` across loops1-5. The archived HTML therefore
shows genuine recurrent correction, not only a better loop1 operating point.

## 8. Decision And Lessons

Keep direct Euclidean address residual as the first successful generic binding
mechanism in this sequence. It passes CE, systems, activity, and real-loop-
correction checks, and beats normal GDN2 by `+0.0807` mean hard blank.

It narrowly misses the preregistered primary quality threshold `+0.10`, so an
unchanged step9300 continuation is not authorized. Exact remains zero, and the
result is not yet a paper-level win.

The remaining high-value causal difference from pure position-Q/K is that the
current candidate forces reads and writes to share one address map. That
creates a symmetric address similarity, while ordinary Q/K and the successful
position diagnostic use independent read/write maps. The next one-shot test
decouples only those two residual projections. It is not a width/rank/scale or
training-duration sweep.

## 9. Submission

Not applicable.
