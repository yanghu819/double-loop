# P-GDN3-015: Bi-Axis Grouped Value Decay

## 1. Metainfo

- Status: approved; not launched
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

Pending strict CUDA contract and exact-resume production probe. No formal
continuation is authorized until both pass.
