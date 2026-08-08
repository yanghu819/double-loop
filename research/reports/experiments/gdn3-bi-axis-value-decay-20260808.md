# P-GDN3-015: Bi-Axis Grouped Value Decay

## 1. Metainfo

- Status: discarded at the exact step3001 production stability gate
- Date: 2026-08-08
- Branch: `codex/gdn3-bi-axis-value-decay-20260808`
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

The pinned official GDN2 state is K32xV32 per head. Its decay and erase act on
K rows. The V-side write gate scales the current payload, but after a payload
is stored all V channels in one K row inherit the same future lifetime. This is
a structural asymmetry: address rows have persistent lifetime control while
value channels only have write-time amplitude control.

P005 changed erase/write coherence, P006 added an independent state, P007 read
incoming state before the scan, P008 added a post-scan transition, P009/P010
changed call composition, P011 routed current writes across heads, P012 changed
the signed K transition spectrum, P013/P014 expanded address capacity, and the
FutureSeed candidates changed inter-layer state transport. None adds a second
persistent forgetting axis inside the original KxV state and original scan.

## 3. Mechanism

For one head, augment the parent recurrence with a diagonal V-axis decay:

`S_t = A_t D^K_t S_{t-1} D^V_t + k_t (w_t * v_t)^T`.

Use eight fixed V groups of four channels. A bias-free D256->64 projection
predicts eight group logits per head. Map each logit to a nonpositive log-decay
with an exact zero point:

`g^V_t = clamp(log(2) - softplus(r_t), max=0)`.

The projection is zero-initialized, so `r_t=0`, `g^V_t=0`, and the parent is
the exact migration function. The map has a direct first-order gradient at the
zero boundary, which is a binding CUDA-contract assertion.

For the fixed 81-token scaffold, retain one unchanged official `chunk_gdn2`
call via the exact moving frame

`C_t = exp(sum_{j<=t} g^V_j)`,

`v'_t = v_t / C_t`,

`o_t = o'_t * C_t`,

`S_T = S'_T * C_T`.

All products are elementwise on the V axis. The official call still receives
the original Q/K/K-decay/erase/write tensors and owns the recurrent backward
graph. The candidate adds exactly `12 * 256 * (8 * 8) = 196,608` parameters,
no recurrent state values, no tokens, no scan, no second core, no reverse
traversal, and no Sudoku-specific operation.

This implementation is an 81-token mechanism test, not a production claim for
unbounded contexts. A positive result would justify a separately audited
chunk-local anchored moving frame inside FLA; a negative result closes this
axis without a Triton rewrite.

## 4. Falsifiable Prediction

If V-channel lifetime interference is the missing recurrent bottleneck, all 12
projections should activate after exact resume, use different group lifetimes
across boards and tokens, remain numerically bounded, and improve late-loop
board closure within 100 matched steps. The primary signal is at least `+0.02`
hard51-64 macro loop5 exact, not CE or blank accuracy alone.

If grouped decay activates and changes the moving frame but exact does not
improve, or if useful activation requires an unstable inverse frame, then a
second persistent axis is not the missing closure mechanism at this parent.
Close it after this candidate; do not tune group count, parameterization,
initialization, scale, seed, LR, loss, batch, width/depth, or duration.

## 5. Migration And CUDA Contract

Before any continuation, exact pushed source in a clean detached worktree must
pass all of:

1. CUDA index0 and UUID exactly match the registered GPU1, with no GPU2 and no
   concurrent compute application;
2. pinned FLA source SHA is
   `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. all 12 layers are official `GatedDeltaNet2` and expose exactly one
   `ChunkGDN2FunctionBackward` each, with no fallback;
4. exact parameter delta is 196,608 and state/token/scan/core deltas are zero;
5. zero projections give bit-exact full output and all 12 terminal states,
   including synthetic nonzero incoming states;
6. all 12 projections receive finite nonzero first-order gradients at zero;
7. an opened synthetic path matches a direct Bi-Axis recurrence for output and
   final state within fixed BF16/FP32 tolerances;
8. grouping is exactly eight groups of four V channels, and opened decay is
   nonpositive, causal, content dependent, and V-group equivariant;
9. cumulative scale stays in `(0,1]`, inverse scale is at most 4, terminal RMS
   is finite and at most `4x` the parent, and no NaN/OOM/fallback occurs;
10. exact-resume model/optimizer/RNG/data-order migration writes a complete
    step3001 checkpoint and metrics from the registered parent.

Any miss closes this implementation before science. It does not authorize a
group count, map, epsilon, clipping, initialization, precision, kernel, or
training-setting rescue.

## 6. Step3001 Production Probe

The one-step probe is migration and production-fit evidence only. It must show
exactly 12 enabled paths, mean absolute grouped log-decay and write-frame
relative change each at least `1e-4`, finite nonzero group/board/token
variation, cumulative scale minimum at least `0.25`, inverse frame at most 4,
terminal RMS at most `4x` the matching parent, unchanged official-kernel
provenance, and complete source/config/checkpoint/metrics hashes.

Probe exact/blank values are diagnostics only and cannot pass the science gate.

## 7. Science, Stability, And Cost Gates

At step3100, activation and stability require all of:

- exactly 12 adapters active;
- mean absolute V log-decay and write-frame relative RMS finite and `>=1e-4`;
- finite nonzero group, board, and token variation;
- cumulative scale finite in `[0.25,1]`, hence inverse frame at most 4;
- finite terminal-state RMS at every loop and at most `4x` frozen control;
- one pinned-official recurrent call and no fallback.

Quality passes by exactly one route:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with every hard
   range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

The original state and call count remain unchanged. Independently warmed
elapsed overhead is capped below 15%, and peak allocated-memory overhead below
10%, versus the frozen control. Timing instability or either ceiling miss
kills the run regardless of quality.

## 8. Required Readout

Report and archive mixed and official51-55/56-60/61-64 loop1-5 exact/blank/
wrong cells; train CE and same-board correction; grouped log-decay magnitude,
active fraction, group/board/token variation, cumulative scale, inverse-frame
bound, write/output/state frame changes, weight RMS and terminal-state geometry;
independently warmed throughput, allocation/reservation and timing; config,
source, parent, checkpoint, metrics and log hashes; and same-board loop1-5
visualization.

## 9. Decision

Discard before formal continuation. Source
`f32e5cbcb206b2b95c36d4555904d6ccd6913759` was pushed and read back, and the
GPU1 worktree was clean and detached at that exact SHA.

Strict CUDA contract R2 completed with status0. It verified the registered UUID,
pinned FLA source SHA, 12 official `GatedDeltaNet2` layers, exactly one
`ChunkGDN2FunctionBackward` per layer, exact 196,608-parameter insertion,
bit-exact zero-init full output and all 12 terminal states including nonzero
incoming states, finite nonzero direct gradients in all 12 projections, the
direct Bi-Axis recurrence reference, eight-group equivariance and synthetic
scale bounds. Contract log SHA256 is
`c3318660bc90058bf4834e194555ef2cb602e631ca60da1567bfe13d2555863d`.
R1 used the repository helper venv, which has no PyTorch, and exited before
model construction or CUDA work. It is separately classified as a non-science
orchestration failure; abort SHA256 is
`28830deaa94ddbc133accc955253641a3f977155137957f80fdbf6b619d38cc5`.

The exact-resume production probe
`p-gdn3-015-bi-axis-value-decay-probe-s3001-20260808T031414Z-f32e5cb`
completed with status0 from the registered step3000 parent. It saved a complete
step3001 checkpoint and metrics, retained the official kernel path, and returned
GPU1 to zero memory with no NaN, OOM, fallback, source drift, checkpoint drift
or concurrent model process.

The path was not dead. At loop5, mean absolute grouped log-decay was
`0.0325488`, active fraction `0.474091`, group/board/token standard deviations
were `0.0111240/0.0004223/0.0012784`, and projection weight RMS was
`0.00140234`. The registered moving-frame stability gate nevertheless failed
decisively:

| Probe readout | Required | Observed |
|---|---:|---:|
| cumulative scale minimum | `>=0.25` | `0.0002853` |
| inverse scale maximum | `<=4` | `3505.29` |
| write-frame relative RMS | finite, active | `198.57` |
| output restore relative RMS | finite | `0.997583` |
| state restore relative RMS | finite | `0.998929` |
| terminal-state RMS | finite, `<=4x` parent | `4.35444` |

The result separates two claims. The Bi-Axis recurrence and its exact parent
embedding are algebraically valid under the strict contract, so persistent
V-channel lifetime remains a coherent architectural concept. This global
81-token moving frame is not a production-stable parameterization: a single
optimizer step makes the inverse frame thousands-fold and nearly rewrites the
entire restored state. Probe exact/blank values are intentionally not used as
science evidence, and the registered step3100 quality/cost test was not run.

Artifact SHA256 values:

| Artifact | SHA256 |
|---|---|
| metrics JSON | `07dee45d2f33f391919fbb686cb9888a3866b434dbe31c229c7a6db4de606f55` |
| step3001 checkpoint | `3d3d9b4de576f4da918f556d17639e559688eec9d1cafc3ab0912913a19b2e8b` |
| config | `ffbfdb40b9a85b4e6f3756d0b42f544efda7b02531d329368e5da80fcdab59ad` |
| run log | `e895cbbfb32925fdf338623b32408c424489a6ae55d8c38353eafbfe29ad0444` |
| launch log | `9978323d1a96b54870df8768290b7f27a1e753281195b9a3f11e9a33e0391269` |
| source snapshot | `342985c767723c284a72fe1c08f0791dc021736925bcb4015cdbee4a8d12ae37` |
| visualization HTML | `60bb8e66e7e675562083153a8df54e538fd788bd14148657266be2b4a0df987f` |
| stability abort | `4fed3082f2aeb175629670b3f526612c256453fecb6a4a2386797522481ffd59` |

Close this implementation without a group-count, map, epsilon, clipping,
initialization, precision, kernel, scale, seed, LR, loss, batch, width/depth or
duration rescue. A future Bi-Axis revisit would need a new intrinsically bounded
chunk-local transition, not a tuned version of this global inverse frame.
