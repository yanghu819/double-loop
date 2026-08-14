# P-GDN3-034: Per-Layer Log-SPD Sudoku Transfer

## 1. Question

Does the independently learned bounded Q/K geometry that improved directional
MQAR also improve hard-Sudoku closure when inserted into each position-QK GDN2
layer without changing FutureSeed, recurrent state, or the official scan?

## 2. Motivation

P020 raised directional-MQAR L1024 balanced accuracy from `0.1735` to
`0.48225` with one independent bounded Log-SPD metric per layer. Its original
absolute gate relied on a historical carrier that is no longer reproducible.
P033's new same-process native control reached `0.17325`, independently
reproducing P020's control, while its cross-layer shared metric regressed to
`0.1250`. Together these results isolate the useful hypothesis: layer-specific
address geometry, not a shared namespace, may reduce binding interference.

This experiment transfers the already-tested mechanism. It does not rerun or
tune MQAR, share parameters, decouple erase/write keys, expand K/V, add a
cache, or add task-specific logic.

## 3. Intervention

For every one of the 12 D256/H8/K32 position-QK GDN2 layers, learn a symmetric
trace-free generator `A` and apply

```text
C = exp(0.5 * log(2) * A / (1 + ||A||_F))
q' = q C
k' = k C
```

before the unchanged pinned official `chunk_gdn2`. The residual implementation
uses `q + q(C-I)` and `k + k(C-I)`, so zero parameters are exactly the parent
function in storage dtype. Each layer owns an independent metric. The metric
eigenvalues remain in `[0.5, 2]` and its determinant remains one.

The candidate adds exactly 50,592 parameters, no recurrent state, no cache and
no scan. Native terminal FutureSeed, random traversal, data, optimizer, RNG,
BF16, effective batch128 and seed52 remain unchanged.

## 4. Falsifiable Prediction

If layer-specific address anisotropy is a transferable binding bottleneck, all
12 metrics will activate and hard51-64 macro loop5 exact will improve by at
least `.02` over the frozen control while every official blank range regresses
by at most `.01` in blank accuracy.

The alternate pass is mixed loop5 exact `+.03`, 61-64 non-regression, and
stronger same-board loop3->5 wrong-cell correction. An active bounded metric
that misses both routes falsifies the transfer at this training horizon.

## 5. Integrity Contract

Before science, require one visible approved GPU at CUDA index0, the pinned FLA
source SHA `9c8e42e...`, a clean detached pushed source, and the exact parent
checkpoint SHA `6339c3cb...`.

The CUDA checker must prove exact +50,592 parameters, zero state/scan delta,
12 independent parameter storages, exact zero-init full output and all terminal
states, exact nonzero-incoming-state identity in every layer, finite nonzero
gradient in every metric, 12 official `ChunkGDN2FunctionBackward` paths,
position-QK and FutureSeed identity, head-permutation equivariance, bounded
eigenvalues/logdet, and a nonzero opened output dependency. The step3001 probe
must prove exact resume, optimizer/RNG/data-order migration, activation and
production fit; it is not a quality score.

## 6. Budget And Kill Rule

After contract and one step3001 probe, run exactly one candidate continuation
from step3000 to step3100. Reuse the frozen control; do not repeat it. Require
training elapsed and independently warmed throughput overhead below 25%, and
peak allocation overhead below 10%.

Any integrity, activation, quality or cost miss closes the mechanism. Do not
rescue metric scale, rank, sharing, seed, LR, loss, batch, width, depth or
duration.

## 7. Results

Pending.

## 8. Decision

Pending the strict CUDA contract, step3001 migration probe, and single formal
candidate.

## 9. Provenance

- MQAR source evidence: P020 per-layer Log-SPD and P033 shared Log-SPD.
- Frozen control: `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`.
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`.
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`.
- Source/run/checkpoint hashes: pending.
