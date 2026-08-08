# P-GDN3-016: Gauge-Balanced Bi-Axis GDN

## 1. Metainfo

- Status: preregistered; implementation and CUDA contract pending
- Date: 2026-08-08
- Branch: `codex/gdn3-gauge-balanced-bi-axis-20260808`
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

## 8. Required Readout

Report mixed and official51-55/56-60/61-64 loop1-5 exact, blank accuracy and
wrong cells; train CE and same-board correction; potential, common K funding,
physical V decay, active fraction, group/board/token variation, frame min/mean/
max, inverse bound, write/output/state frame changes, weight RMS and terminal
geometry; independently warmed throughput, allocation/reservation and timing;
config, source, parent, checkpoint, metrics and log hashes; and same-board
loop1-5 visualization.

## 9. Decision

Pending strict CUDA contract and exact step3001 production probe. Formal
step3000-to3100 continuation is forbidden until both pass.
