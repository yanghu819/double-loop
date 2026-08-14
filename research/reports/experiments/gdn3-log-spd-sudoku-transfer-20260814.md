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

Strict R6 completed status0 from exact pushed source
`0ee1aac63237308c027972acf00a323a8d69a8d2` on CUDA index0, UUID
`GPU-31166d8c-9fe5-d953-dc44-d0d549969ada`. It verified all 12 pinned-official
GDN2 backward paths, exact +50,592 parameters, zero state/scan delta,
independent per-layer storage, zero-init full output/state identity, arbitrary
nonzero incoming-state identity, direct finite gradients, head permutation,
bounded geometry, and no fallback. Contract log SHA256 is
`81c5dd9bf78ea3fe1432aea9d8a36a46e299eeffbd3aeab3bd47125bf3a510c1`.

The exact step3001 probe also completed status0. Its metric raw RMS is
`0.001402`, metric delta is `0.034659`, eigenvalues are
`[0.9677,1.0333]`, and condition number is `1.053`; this establishes migration
and activation, not quality. Probe metrics/checkpoint SHA256 are
`1b9a07e8b13320086e658dc343a5a8d6ff908828c154be2bc9e2dd4082072de1`
and `89013cc7270d55becd5326df8705fafb43d3980ed986def44cfc407d64b07198`.

The formal run
`p-gdn3-034-log-spd-s3100-20260814T090743Z-0ee1aac` completed status0. All 12
metrics remain active at loop5: raw RMS `0.015537`, metric delta `0.259015`,
eigenvalues `[0.741314,1.418861]`, condition `1.617079`, and maximum absolute
logdet `0.163689`. Train CE improves slightly from control `0.858617` to
`0.857107`.

Quality does not transfer. Hard51-64 macro loop5 exact is unchanged at
`0.000651`; mixed loop5 exact regresses `0.025391->0.023438`. Official loop5
51-55/56-60/61-64 exact remains `0.001953/0/0`, while blank accuracy changes
as follows:

| Range | Control blank | Candidate blank | Delta |
|---|---:|---:|---:|
| 51-55 | 0.573766 | 0.574410 | +0.000644 |
| 56-60 | 0.503861 | 0.501596 | -0.002265 |
| 61-64 | 0.591923 | 0.597692 | +0.005769 |

On the same 256 boards per range, loop3-to5 mean wrong-cell correction changes
from `-0.015625->0.089844` on 51-55, `0.066406->0.035156` on 56-60, and
`0.183594->0.210938` on 61-64. Thus the candidate improves late correction in
two ranges but weakens it in 56-60, failing the alternate route.

Effective continuation throughput falls `15.4969->9.3366` boards/s. Raw
elapsed overhead is `+65.98%`, above the fixed 25% ceiling; peak allocated and
reserved memory overhead are `+3.99/+4.07%`, within budget. No NaN, OOM,
fallback, source drift, data drift, or concurrent GPU process occurred.

## 8. Decision

Discard. The per-layer Log-SPD mechanism is trainable, bounded, and causes a
small favorable blank-accuracy shift on the hardest range, but does not improve
full-board closure and regresses mixed exact. The failure is not inactivity or
state instability. It falsifies transfer of P020's relative MQAR gain to this
Sudoku parent and closes scale, rank, sharing, seed, LR, loss, batch, width,
depth, and duration rescue. The expensive per-layer matrix exponential also
misses the registered elapsed ceiling despite its small parameter and memory
footprint.

## 9. Provenance

- MQAR source evidence: P020 per-layer Log-SPD and P033 shared Log-SPD.
- Frozen control: `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`.
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`.
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`.
- Formal source SHA: `0ee1aac63237308c027972acf00a323a8d69a8d2`.
- Formal metrics SHA256:
  `5f4531a1ad3d872e92c5382e52082ac3c58c0302683b8d51005713208793fc22`.
- Formal checkpoint SHA256:
  `a979403eafcc467a96e3b3144319cb2446ee40dbac00854a599b390b090950f7`.
- Config/log/source-snapshot SHA256:
  `6f56247108d1ebc0771419370dbe6c9f79bd7ed14b6693e25413d3d82831deb9`,
  `9b52ce4e15f8fd1df7d0e3fc213ce46105910738af1e27cae432f9be1e57cb01`,
  `0aae8b4f906be58abf44ca02cc8f907e8044425daa0901ad444ce7d468413b2f`.
- Comparison directory:
  `/huyang2/double-loop/runs/p-gdn3-034-comparison-20260814T093835Z-0ee1aac`.
- Comparison/abort/visualization SHA256:
  `912a7baff3beb2ca00ff4d9d1df922fe11ea645c6cb5c361afdb6d4a1a85adec`,
  `31cf18f3805feaff7b2bd7c23ab9367551373207dbde0f585c8262726facc06c`,
  `72b7f6ccd22b4184c2e4892bf42533cc771cfe1b99a8288164d8ed114bafd457`.
