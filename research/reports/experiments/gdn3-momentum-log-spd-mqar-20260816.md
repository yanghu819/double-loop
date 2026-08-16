# P-GDN3-060: Momentum Delta With Bounded Coherent Address Geometry

## 1. Research Question

Can a bounded learned address geometry remove P059's adjacent-owner tail while
preserving the second-order recurrence and native Momentum FutureSeed that
already solve most L1024 retrieval?

## 2. Evidence And Mechanism

P059 reaches `.94425` balanced accuracy but leaves 223 errors, including 151
valid values assigned to the adjacent write owner. The frozen component audit
shows that transporting only momentum reproduces P059 within `.0005`; the
remaining errors therefore arise inside each token scan, not from selecting
the wrong FutureSeed component.

P060 retains the exact P059 Momentum Delta recurrence, stacked `[S,M]` state
and native cross-layer transport. Each layer adds one head-local bounded
Log-SPD factor and applies it coherently to both Q and K after the external
Triton short convolution and before the external recurrence normalizes them.
Read, write and error prediction continue to use one shared owner key. The raw
metric is zero initialized, so the parent function and every nonzero incoming
state are exactly P059 at launch.

This is not the failed direct decouple-key family. P031/P036 separate erase and
write coordinates and lose their common owner anchor. P060 never introduces a
second key. It composes two independently positive mechanisms: P059's live
second-order transition and P020's per-layer bounded coherent geometry. The
composition is falsified unless it specifically reduces the residual adjacent
wrong-owner tail.

## 3. Fixed Protocol

- directional MQAR L1024, four pairs, D128/L2/H4/K32/V32;
- 10 epochs, batch32, seed123, identical fixed data and initialization;
- external Momentum DeltaNet exact SHA
  `c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`;
- frozen P059 score and cases are the only control; no control rerun;
- candidate adds exactly 4,216 parameters, 2,108 per layer;
- recurrent state stays exactly 8,192 values per layer `[S,M]`;
- no Sudoku logic, selector, cache, replay, extra scan or new state;
- no sweep.

## 4. Falsifiable Prediction

If local address anisotropy causes the last owner collisions, a small coherent
metric should retain both directional accuracies while removing at least one
fifth of P059's errors and adjacent wrong-key swaps. If accuracy merely moves
without reducing the conditional wrong-key share, the composition does not
solve ownership and is closed.

## 5. Registered Gates

Integrity and activation:

- one registered CUDA index 0 and exact GPU UUID;
- exact pushed clean detached source and frozen artifact hashes;
- exact external SHA/source hashes and two Momentum chunk backward paths;
- full-model and per-layer nonzero-state bit-exact P059 identity at raw zero;
- exact +4,216 parameters and unchanged `[S,M]` state geometry;
- finite nonzero gradients for both metrics, Q/K/V, momentum and FutureSeed;
- both metrics have factor delta Frobenius mean `>=1e-4` after training;
- actual metric eigenvalues remain positive, condition `<=4.1`, and absolute
  BF16 log determinant `<=.05`;
- two active bounded Momentum layers and one native FutureSeed route, with
  `M/S` RMS ratio in `[.01,20]`.

Quality relative to frozen P059:

- balanced accuracy `>=.955` and gain `>=.01`;
- future and past accuracy each regress by at most `.005`;
- joint exact `>=.84` and gain `>=.01`;
- total errors `<=178` and at least 20% below 223;
- wrong-key swaps `<=120` and at least 20% below 151;
- conditional wrong-key share improves by at least `.05` from `.677130`;
- adjacent-write-rank swaps fall by at least 20%.

Cost relative to frozen P059:

- elapsed, post-warm wall and warmed-step ratios each `<1.30x`;
- peak allocation ratio `<1.10x`.

Any miss closes this exact composition. There is no metric cap, scale, rank or
sharing rescue; no momentum coefficient, gate or state-size rescue; and no
seed, LR, loss, batch, width, depth or duration sweep. Sudoku transfer is
allowed only after every gate passes.

## 6. Status

Exact pushed/read-back source
`d4fd8003e685dbad02dd82e7352f98d4d5c0ac07` stopped at the strict contract,
before formal training. The synthetic opened metric is active and coherent:
address relative RMS is `.03393`, head-permutation error is zero, FP32 metric
condition is `2.0592`, and FP32 absolute log determinant is `4.46e-6`.

The actual BF16-applied factor fails the preregistered volume-stability gate.
Its condition remains bounded at `2.0628`, but absolute log determinant is
`.06724`, above the fixed `.05` ceiling. Dense elementwise BF16 rounding thus
turns an analytically volume-preserving factor into roughly 7% volume drift in
the effective address transform.

Close P060 without formal training. Do not lower the synthetic opening, relax
the threshold, change metric cap/scale/rank/sharing, or rerun. The launcher
wrote a non-science contract `abort.json`; no checkpoint or score exists.

Across 34 sampler rows, active-memory samples average `53.43%` GPU utilization
and peak at `85%`; sampled memory peaks at `3,022 MiB` and power at `204.31 W`.

Artifacts:

- run: `/huyang2/double-loop/runs/p-gdn3-060-momentum-log-spd-20260816T001911Z-d4fd800`
- contract log SHA256: `4540c8fffeada5cd8f0020764587c8785925876d7530a973ca40452e05ed571d`
- GPU samples SHA256: `41a606a563294ee1ecf857fc6e848497a51ee78038916e77290fd17b59910a61`
- source snapshot SHA256: `a4367a7b411cd6403a62793413ddeddb02d38337606ca892e0badb1378328f6f`
- abort SHA256: `1051333aff19fee5b0e87e594bb2dbaa40b3d2fbee7f8f5cb5c9898ef2bea7a1`
