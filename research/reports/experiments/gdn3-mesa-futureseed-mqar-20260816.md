# P-GDN3-066 Official Mesa Least-Squares FutureSeed

## 1. Metainfo

- Plan: `P-GDN3-066`
- State: approved, implementation/contract pending
- Decision field: directional MQAR L1024/K4, not Sudoku
- Resource: one AIStation A800 80GB, CUDA index 0 only
- Frozen reference: `P-GDN3-059` Momentum DeltaNet + native `[S,M]` FutureSeed

## 2. Hypothesis

P059's remaining errors are predominantly correct-value/wrong-key swaps. All
151 swaps use an adjacent write rank and 150/151 preserve direction. More
slots, value width, write controllers, dual keys, side memories, phase mixing,
and pre/post-scan residuals have failed. The missing operation is therefore not
more storage but an address-interference-aware read.

Mesa stores decayed key Gram and key-value sufficient statistics and reads by
solving a regularized normal equation. This should separate correlated keys
without a Sudoku selector or another learned wrapper. Native FutureSeed carries
the complete `[Hkk,Hkv]` state to the receiver with one positive joint scale,
preserving their relative units.

## 3. Fixed Configuration

- Arm: `future_seed_mesa`
- Model: D128/L2/H4/K32/V32, output gate enabled
- Recurrence: exact pinned official FLA `chunk_mesa_net`
- State per layer: `Hkk[H,32,32] + Hkv[H,32,32] = 8,192` values
- Solver: 30 conjugate-gradient iterations, fixed official default
- Regularizer lower bound: `0.25`, fixed official default
- FutureSeed: one adjacent-layer route, joint RMS normalization of `[Hkk,Hkv]`
- Data: fixed directional MQAR L1024/K4, 10,000 train and 1,000 test examples
- Train: 10 epochs, batch32, seed123, LR/WD inherited unchanged
- Initialization: matched compatible shell tensors from the frozen parent;
  Mesa recurrence parameters train from scratch because this is a foundational
  recurrence, not a 100-step zero-init graft
- Sweep: none

## 4. Environment and Provenance

- FLA marker: `9c8e42e762fce087c27b673af4922795d9edb85e`
- Official source: `fla/layers/mesa_net.py` and `fla/ops/mesa_net/*`
- License: MIT; no external source is copied into this repository
- Runtime: pinned PyTorch 2.8 / Triton >=3.4 environment
- Launch requires exact pushed SHA, GitHub readback, clean detached worktree,
  exact parent/frozen-artifact hashes, target GPU UUID, and no other compute app

## 5. Contract and Commands

The strict contract must prove:

1. Exact pinned source paths and file hashes.
2. Exactly two `ChunkMesaNetFunctionBackward` paths and no fallback.
3. Finite nonzero gradients for Q/K/V, decay, write, lambda, output, output
   gate, both initial states, producer terminal state, and the receiver
   FutureSeed gate.
4. Official chunk output/state parity against `naive_mesa_net_exact` with a
   nonzero initial state: output relative RMS <=0.05 and both state relative
   RMS <=0.01.
5. Head permutation equivariance <=0.005 relative RMS.
6. Exact parameter/state accounting, symmetric PSD Hkk up to 0.01 numerical
   tolerance, finite board variation, and endpoint CG relative error <=0.05.

Launch command:

```bash
setsid env EXPECTED_SOURCE_SHA=<pushed-sha> \
  bash scripts/run_zoology_mesa_futureseed.sh
```

## 6. Artifacts

Pending contract and formal run.

## 7. Registered Decision Gates

Quality must beat P059 rather than merely recover old GDN2:

- balanced accuracy >= P059 +0.01 (`>=0.95425`)
- future and past accuracy regress by at most 0.005
- joint exact >= P059 +0.01 (`>=0.834`)
- total errors <=180
- wrong-key valid-value swaps <=105
- wrong-key swap share <=0.60713

Cost and stability:

- elapsed, post-warm wall, and warmed-step ratios each <=3.0x P059
- peak allocation <=2.0x P059
- two active Mesa layers, exactly one active FutureSeed route
- finite nonzero state/board variation, stable Hkk geometry, CG error <=0.05

Any integrity, activation, stability, quality, or cost miss closes this exact
Mesa configuration. No CG/lambda/output-gate/FutureSeed-gate/seed/LR/loss/
batch/width/depth/duration rescue is authorized.

## 8. Conclusions

Pending. A pass establishes a better GDN3 recurrence, after which one matched
state-component isolation is required to determine whether the joint
`[Hkk,Hkv]` transport is also a better FS2. A fail closes least-squares Mesa at
this fixed condition and returns the search to a different scalable recurrence.

## 9. Submission Record

Not applicable.
