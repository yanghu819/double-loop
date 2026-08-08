# P-GDN3-016: Gauge-Balanced Bi-Axis GDN

## 1. Metainfo

- Status: discarded after complete matched endpoint
- Date: 2026-08-08
- Branch: `codex/gdn3-gauge-balanced-bi-axis-20260808`
- Formal source SHA: `79469ed71e43c5af0175adf831e56487dc8f581c`
- Formal run:
  `p-gdn3-016-gauge-balanced-s3100-20260808T043751Z-79469ed`
- Matched comparison:
  `/huyang2/double-loop/runs/p-gdn3-016-comparison-20260808T050000Z-79469ed`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent: D256/L12/H8/K32/V32 position-QK GDN3 plus native terminal FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen control: `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Evidence Boundary

P-GDN3-015 proved that grouped persistent V-axis decay has an exact reduction
to one unchanged official GDN2 call and a direct first-order learning path. It
also proved that a sequence-global cumulative frame is unusable: after one
optimizer step its minimum scale was `0.0002853`, inverse scale was `3505.29`,
and write-frame relative RMS was `198.57`.

The failure is numerical parameterization, not an algebraic contradiction.
The supplied Bi-Axis proposal explicitly warns against a global cumulative
frame and requires a local or otherwise bounded anchor. Externally splitting
the official call at token64 is not repeated: P-GDN3-009 already established
that this execution does not retain the registered cross-call recurrent
backward graph. A valid successor must therefore keep one official call while
bounding both coordinate directions intrinsically.

## 3. Mechanism

Use the same eight fixed V groups and zero-initialized bias-free D256-to-H8xG8
projection as P015, but predict a bounded value-coordinate potential instead
of integrating a negative hazard from the sequence start:

`phi_t = log(4) * tanh(r_t)`.

Let `delta_t = phi_t - phi_(t-1)`, with `phi_(-1)=0`, and define one nonnegative
common contraction per token and head:

`m_t = relu(max_g delta_(t,g))`.

The physical grouped V-axis log-decay is

`gV_(t,g) = delta_(t,g) - m_t <= 0`.

The common factor is funded by the existing K decay:

`gK'_t = gK_t - m_t <= gK_t <= 0`.

With `C_t=diag(exp(phi_t))` and transformed state `S'_t=S_t C_t^-1`, the
physical recurrence

`S_t = A_t D^K_t S_(t-1) D^V_t + k_t u_t^T`

reduces exactly to standard GDN2:

`S'_t = A_t exp(gK'_t) S'_(t-1) + k_t (u_t / exp(phi_t))^T`.

Thus the implementation transforms V writes by `exp(-phi_t)`, calls the same
pinned official `chunk_gdn2` exactly once, and restores outputs and the final
state by `exp(phi_t)`. The frame and inverse are analytically bounded by four.
The physical V decay is always nonexpansive. Cumulative relative lifetime
between any two groups equals their potential difference, allowing up to a
16x relative lifetime ratio without an unbounded numerical frame.

The candidate adds exactly 196,608 parameters and no recurrent state values,
tokens, scans, second core, reverse traversal, or Sudoku operation.

## 4. Falsifiable Prediction

If P015 failed only because of the global coordinate frame, P016 should retain
all 12 direct gradients, activate group-dependent physical V decay after exact
resume, keep frame and inverse at most four by construction, and avoid the
one-step state rewrite. Within 100 matched steps, this stable extra lifetime
axis should improve board closure rather than merely CE or blank accuracy.

If the stable path activates but exact does not improve, then persistent
V-axis lifetime is not the missing closure mechanism at this parent. If the
path cannot pass exact identity, direct recurrence, nonexpansion, gradient, or
one-step production stability, close this parameterization before science.
Do not tune the potential cap, groups, common-shift rule, seed, LR, loss, batch,
width, depth, or continuation length.

## 5. Migration And CUDA Contract

Exact pushed source in a clean detached worktree must pass all of:

1. CUDA index0 and the registered GPU1 UUID, with no concurrent compute app;
2. pinned FLA source SHA
   `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. 12 official `GatedDeltaNet2` layers and exactly one
   `ChunkGDN2FunctionBackward` per layer;
4. exact parameter delta 196,608 and zero state/token/scan/core delta;
5. bit-exact zero-init full output and all 12 terminal states, including
   synthetic nonzero incoming states;
6. finite nonzero first-order projection gradients in every layer at zero;
7. opened moving-frame output/final state matches a direct physical Bi-Axis
   recurrence within fixed BF16/FP32 tolerances;
8. physical V log-decay and modified K log-decay are nonpositive;
9. frame scale and inverse stay in `[0.25,4]`, grouping is eight groups of four
   channels, and the transition is causal and group-permutation equivariant;
10. cumulative relative group lifetime equals the bounded potential difference,
    terminal RMS is finite and at most `4x` parent, and no fallback occurs.

Any miss closes P016. It does not authorize a cap, map, max operator, group,
precision, initialization, or training-setting rescue.

**Result:** passed on the clean detached pushed SHA. The contract verified the
registered GPU UUID and pinned FLA SHA, exactly 12 official GDN2 layers and one
`ChunkGDN2FunctionBackward` per layer, exact `+196,608` parameters, bit-exact
zero-init full output and all 12 terminal states including nonzero incoming
states, finite nonzero gradients in all 12 projections, nonpositive physical
V and modified K decay, direct physical-recurrence agreement, causal/group
equivariance and the analytic frame bound. Contract log SHA256:
`aa0719eeb3bfc85268ea8dc6a0f2498bd390a1306788104be036c96e74fcafbf`.

## 6. Step3001 Production Probe

The exact-resume probe is migration and production-fit evidence only. It must
write complete metrics/checkpoint/config/source hashes and show:

- exactly 12 enabled paths;
- potential abs, physical V log-decay abs, and write-frame relative RMS each
  finite and at least `1e-4`;
- common K contraction finite and at least `1e-6`;
- finite nonzero group, board, and token variation;
- frame minimum at least `0.25`, frame maximum and inverse maximum at most `4`;
- finite terminal RMS at every loop and at most `4x` the matching parent;
- unchanged official-kernel provenance and no NaN/OOM/fallback.

Probe exact and blank scores cannot pass the science gate.

**Result:** passed on exact resume step3000-to3001. At loop5, potential,
physical V decay, common K contraction and write-frame relative RMS were
`0.079235/0.128195/0.128196/0.091672`; group/board/token variation was
`0.000364/0.007307/0.025817`. Frame min/max/inverse was
`0.716037/1.391910/1.396575`, and terminal RMS was `1.186997`. Metrics,
checkpoint, probe log and verdict SHA256 were respectively
`5caf5a0f9736562085ca93740a82551d89bffaed8eef8d44e00caa445d6a1f41`,
`ef7e11fff1d2d2faa7576ab2406eef2b197b8ccc51fc4e5ff6eb8d9c984554af`,
`c44e68d3a9829dadfd364ce3b5fadd1b548b2ac7041565189f2ed6c2a7d16e8a`,
and `901430e792f0932af44a00616966d3ebca51fb5284f3f9a39870c6827e8bbd7d`.
Four earlier wrappers exited before an official-kernel training step because
of explicit environment/argument assertions; their non-science abort receipts
remain archived and do not alter the production result.

## 7. Science And Cost Gates

At step3100, activation and stability retain the step3001 gates. Quality passes
by exactly one route:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with every hard
   range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

Independently warmed elapsed overhead is capped below 20% and peak allocated
memory overhead below 10% versus the frozen control. Any activation, stability,
quality, timing, memory, or integrity miss discards P016 with no rescue.

**Result:** activation and bounded-geometry gates pass, but both quality routes
and both cost gates fail. Formal loop5 potential/physical-V/common-K magnitude
is `0.564159/0.351812/0.351764`; active fraction and group/board/token variation
are `0.875106/0.002631/0.016930/0.067955`. The frame reaches
`0.250029/1.252074/3.998026` with inverse max `3.999530`: bounded, but nearly
saturated at both limits. Hard51-64 macro exact changes `0.000651->0`; mixed
exact changes `0.025391->0`. Official51-55/56-60/61-64 loop5 blank deltas are
`-0.220564/-0.181016/-0.297518`. Train CE changes
`0.858617->1.498811`.

The same-board bank confirms a trajectory failure rather than sampling noise.
Mean wrong cells at loops1-5 are control/candidate
`25.74/24.35/23.92/23.86/23.93` versus
`36.31/35.09/35.16/35.17/35.17` for 51-55,
`29.70/28.20/28.08/28.04/28.02` versus
`39.33/38.44/38.44/38.44/38.47` for 56-60, and
`31.86/27.20/26.61/26.51/26.43` versus
`46.91/45.39/45.48/45.48/45.47` for 61-64. The candidate is worse on all 256
hardest-range loop5 boards; its loop1-to5 correction is `1.441` cells versus
control `5.426`.

Throughput changes `15.497->12.051` effective boards/s. Elapsed, peak allocated
and peak reserved overhead are `+28.60/+21.55/+20.41%`, versus fixed limits of
`+20/+10%` for elapsed/allocation.

## 8. Required Readout

Report mixed and official51-55/56-60/61-64 loop1-5 exact, blank accuracy and
wrong cells; train CE and same-board correction; potential, common K funding,
physical V decay, active fraction, group/board/token variation, frame min/mean/
max, inverse bound, write/output/state frame changes, weight RMS and terminal
geometry; independently warmed throughput, allocation/reservation and timing;
config, source, parent, checkpoint, metrics and log hashes; and same-board
loop1-5 visualization.

Formal loop1-to5 exact / blank accuracy:

| Slice | Control exact | Candidate exact | Control blank | Candidate blank |
|---|---|---|---|---|
| mixed | 0.017578 / 0.023438 / 0.025391 / 0.025391 / 0.025391 | 0 / 0 / 0 / 0 / 0 | 0.515087 / 0.536904 / 0.544841 / 0.545540 / 0.545610 | 0.321282 / 0.341561 / 0.340687 / 0.340792 / 0.340617 |
| official51-55 | 0 / 0 / 0.001953 / 0.001953 / 0.001953 | 0 / 0 / 0 / 0 / 0 | 0.532345 / 0.566677 / 0.573945 / 0.573623 / 0.573766 | 0.333047 / 0.354026 / 0.352916 / 0.353095 / 0.353202 |
| official56-60 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0.473182 / 0.498061 / 0.501699 / 0.503140 / 0.503861 | 0.309530 / 0.323977 / 0.323599 / 0.322535 / 0.322844 |
| official61-64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0.504319 / 0.578523 / 0.589207 / 0.591954 / 0.591923 | 0.269955 / 0.294710 / 0.295321 / 0.294619 / 0.294405 |

Completed formal artifact SHA256:

- metrics: `56fe5d1dce4c3457eb2ba4b485da2751e2d46930b22dafbd9442a3e32e7108c0`;
- checkpoint: `653a0b1e0d35f35cc74fd064328dba2c3bc5ececc904f75d5abcec0d4167ec67`;
- config: `eff05e173a2e8210069c34626e0a38eefa11860439fa65c9a7d3f24f8a600860`;
- run log: `b3a3dbfb7d226f800ce8b137b4e4dd8acfb7b37415d0ea0da1809a752a0f32fa`;
- source snapshot: `1c1ead0eb6a19a17df681012bb1a36b9fa89c04f18d13c74dacfccd01a374f5d`;
- formal launch log/status:
  `be8be481c2c0a3aa117a0c432fce3ecb545d3643c76ac0e6297e2cb64380f672` /
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`;
- comparison JSON/HTML/manifest:
  `24eac1f094afa6e9da298a622ed925bbcc0af73f4f911e7cad75930b2e342466` /
  `aa97d9ac00abc2339381c684d20d6bcc169926aafcf1daf638a89d37fd9b80e9` /
  `7166ab92112e04c30d921b845b6fa2ed766bcd6a0d779de4734408aa4948d21f`.

## 9. Decision

Discarded. P016 resolves P015's numerical ambiguity: the gauge-balanced
parameterization preserves the parent at migration, trains through one official
kernel call, keeps the moving frame bounded and remains fully active. It still
destroys board closure. The learned potential saturates the coordinate bounds,
while the common K contraction needed to keep physical V decay nonexpansive
aggressively forgets the parent's useful state. This is a scientific failure of
the registered stable lifetime mechanism, not another implementation failure.

Do not rescue potential cap, V groups, common-shift map/max, initialization,
precision, seed, LR, loss, batch, width/depth, or duration. No further GPU run
is authorized by this report.
