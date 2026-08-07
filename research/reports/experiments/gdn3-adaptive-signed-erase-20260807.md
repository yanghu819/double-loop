# P-GDN3-012: Adaptive Signed-Erase Spectrum

## 1. Metainfo

- Status: discarded after clean matched endpoint; no rescue
- Date: 2026-08-07
- Planned branch: `codex/gdn3-adaptive-signed-erase-20260807`
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
- Frozen control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Evidence Boundary

The official GDN2 transition along the normalized key direction has the erase
factor `1-b`. The current parent squashes `b` to `[0,1]`, so this component is
nonnegative and cannot represent an oscillatory or sign-reversing memory mode.
Official FLA explicitly supports `b` up to 2 through its
`allow_neg_eigval` path, but enabling that flag directly would not preserve the
frozen parent at exact resume.

P005 only mixed erase and write gates inside their existing nonnegative range.
P007 added inherited-state-conditioned logits before the same sigmoid, still
leaving `b` in `[0,1]`. P008 added a second nonnegative sweep and exposed
explosive state magnitude. P009/P010 tested call composition, and P011 tested
cross-head payload routing but closed at a production precision gate before a
formal score. None tests whether the live single-scan recurrence needs an
adaptive signed transition spectrum.

## 3. Mechanism

Keep the original 81-token, one-call pinned-official position-QK GDN2 path. In
each layer, apply one bias-free, zero-initialized, head-shared `V32 -> K32`
projection to the current per-head payload:

`r_t = tanh(W v_t)`

`b'_t = clamp(b_t + r_t, 0, 2)`

Then call the unchanged official `chunk_gdn2` with the original Q/K/V/decay/
write tensors and `b'`. At `W=0`, `b'=b` exactly. When trained, the content-
conditioned residual can move selected key channels above one, giving the
live transition a negative key-direction eigenvalue without adding a state,
token, scan, second core, reverse traversal, or task rule.

The projection is shared over the eight heads, so the adapter is equivariant
to head permutation. It adds exactly `12 * 32 * 32 = 12,288` parameters and no
persistent state. The official call count, physical sequence length, QK
address geometry, native FutureSeed transfer and output path remain unchanged.

## 4. Falsifiable Prediction

If monotonically nonnegative erase dynamics prevent GDN2 from cancelling stale
or contradictory partial assignments, the learned path should use both signs:
finite content-dependent erase residuals and a nonzero fraction of `b'>1`.
The primary signature is stronger same-board loop3-to5 correction and at least
`+0.02` hard51-64 macro loop5 exact, not merely lower CE or loop1 accuracy.

If the projection activates but `b'>1` remains unused, state magnitude becomes
unstable, or exact does not improve, signed transition spectrum is not the
missing closure mechanism at this checkpoint. Close it after this one matched
candidate; do not tune residual scale, clamp, source, sharing, seed, LR, loss,
batch, width/depth, or duration.

## 5. Migration And CUDA Contract

Before any continuation, exact pushed source in a clean detached worktree must
pass all of:

1. CUDA index0 and UUID exactly match the registered GPU1, with one compute
   application and no GPU2 visibility;
2. pinned FLA source SHA is
   `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. all 12 layers are official `GatedDeltaNet2` and expose exactly one
   `ChunkGDN2FunctionBackward` each, with no fallback;
4. exact parameter delta is 12,288 and state/token/scan/core deltas are zero;
5. zero projection gives bit-exact full output and all 12 terminal states,
   including synthetic nonzero incoming states;
6. all 12 projection gradients are finite and nonzero at zero initialization;
7. a synthetic opened path changes `b`, output and state, uses values both
   below and above one while staying in `[0,2]`, and has finite gradients;
8. head permutation commutes with the shared adapter to max error `<=3e-6`;
9. exact-resume model/optimizer/RNG/data-order migration completes from the
   registered parent and writes a complete step3001 checkpoint and metrics;
10. no NaN, OOM, source/data drift, hidden CPU model path, concurrent GPU
    process, or Sudoku/traversal-specific dependency.

Any contract or production-probe miss closes the implementation. It does not
authorize a residual scale, clamp, source, nonlinear map, layer/head sharing,
precision fallback, or training-setting rescue.

## 6. Step3001 Production Probe

The exact one-step probe is migration and production-fit evidence only. It must
show all 12 paths enabled, finite nonzero residual relative RMS and board/token/
head variation, finite `b'` in `[0,2]`, unchanged official-kernel provenance,
complete checkpoint/config/source hashes, and terminal RMS no more than `4x`
the matching parent readout. A nonzero negative-eigenvalue fraction is recorded
but is binding only at the formal endpoint, because one optimizer step is not
a mechanism-quality budget.

## 7. Science, Stability, And Cost Gates

At step3100, activation and stability require all of:

- exactly 12 adapters active;
- mean absolute erase residual and erase relative change each finite and
  `>=1e-4`, with finite nonzero board/token/head variation;
- fraction of effective erase channels with `b'>1` at least `1e-3` on the
  fixed mixed loop5 readout;
- effective erase finite and contained in `[0,2]`;
- finite terminal-state RMS at every loop and no more than `4x` the frozen
  control at the matching loop;
- no fallback.

Quality passes by exactly one route:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with every hard
   range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

The original recurrent call and state size remain unchanged. Independently
warmed elapsed and peak allocated-memory overhead are each capped below `25%`
versus the frozen control. Timing instability, OOM, or either ceiling miss
kills the run regardless of quality.

## 8. Required Readout

Report and archive mixed and official51-55/56-60/61-64 loop1-5 exact/blank/
wrong cells; train CE and same-board correction; erase residual, effective
erase min/max, `b'>1` fraction, board/token/head variation and terminal-state
geometry; independently warmed throughput, allocation/reservation and timing;
config/source/parent/checkpoint/metrics/log hashes; and same-board loop1-5
visualization.

## 9. Decision

Discard. Complete source SHA
`c09c36851e5a17224eae9d38a40923c77fe6cba2` was pushed and read back before
the run, and the formal worktree was clean and detached at that exact SHA.
The strict R1 GPU1 contract passed the target UUID, pinned FLA SHA, exact
12,288-parameter delta, one official `ChunkGDN2FunctionBackward` per layer,
zero-init output/state identity including nonzero incoming state, gradients,
head equivariance, opened signed spectrum and `[0,2]` bounds. Contract log
SHA256 is
`6d1badaaa24533646b2f926770c2cc933659726ee19e8b9f42310fa8edb1dbde`.

The exact step3001 production probe completed from the frozen parent and
activated all 12 paths. Its metrics/checkpoint SHA256 values are
`7fdc5fb8998666b8578b602b747fd6a4fcfc134022b1abe3e4e1132f12222639`
and
`564854a3063e2837e6f3ef732f1973ae9cfda6fa91aa84b011d84ce060896aca`.
The formal candidate then exact-resumed the original step3000 parent to
step3100 with status0, no NaN/OOM/fallback, and no concurrent GPU process.

### Endpoint quality

| Readout | Frozen control | Candidate | Delta |
|---|---:|---:|---:|
| hard51-64 macro loop5 exact | 0.000651 | 0.000651 | +0.000000 |
| mixed loop5 exact | 0.025391 | 0.023438 | -0.001953 |
| official51-55 loop5 blank | 0.573766 | 0.573909 | +0.000143 |
| official56-60 loop5 blank | 0.503861 | 0.502385 | -0.001476 |
| official61-64 loop5 blank | 0.591923 | 0.592015 | +0.000092 |
| train CE | 0.858617 | 0.858690 | +0.000073 |

The primary quality route fails because hard macro exact is unchanged. The
alternate route fails because mixed exact loses one of 512 boards and
same-board loop3-to5 correction is not stronger in all hard ranges.

All three official case banks match by data hash and contain the same 256
boards per range. Mean wrong cells across loops1-5 are:

- 51-55 control `25.74/24.35/23.92/23.86/23.93`, candidate
  `25.64/24.25/24.02/23.98/23.92`; loop3-to5 correction
  `-0.016 -> +0.102`;
- 56-60 control `29.70/28.20/28.08/28.04/28.02`, candidate
  `29.65/28.34/27.99/28.00/27.95`; loop3-to5 correction
  `+0.066 -> +0.039`;
- 61-64 control `31.86/27.20/26.61/26.51/26.43`, candidate
  `31.66/27.33/26.64/26.54/26.54`; loop3-to5 correction
  `+0.184 -> +0.098`.

### Activation, stability, and cost

The mechanism is not dormant. At mixed loop5, all 12 paths are enabled;
erase residual absolute/relative RMS is `0.123345/0.303028`, board/token/head
variation is `0.006507/0.011541/0.052904`, effective erase spans
`[0,1.84375]`, and `8.707%` of channels use `b'>1`. Terminal RMS/board std is
bounded at `7.200136/0.402642`. Activation and stability gates pass.

The 100-step continuation takes `963.370s` versus `825.970s`; throughput is
`13.287` versus `15.497` effective boards/s, for `+16.64%` elapsed overhead.
Peak allocated/reserved memory is `13982.6/15102.0 MiB` versus
`13186.4/14288.0 MiB`, or `+6.04/+5.70%`. Both preregistered 25% cost gates
pass.

Formal metrics/checkpoint SHA256 values are
`949ce8078b1cc882f1cd527c50d0ad89ea0abf89d6f9fc519a77cff849aad8f8`
and
`dffb6600c9bd153e68d13ebe5a9c3278ffeeb29b0078f450bdcd21f3c252044f`.
The machine-readable comparison and same-board loop visualization are in
`/huyang2/double-loop/runs/p-gdn3-012-comparison-20260807T090000Z-c09c368`.

This closes adaptive signed erase at this checkpoint. Signed transition modes
are learnable, active, stable and affordable, but they do not provide the
missing global closure. Do not rescue projection source, map, clamp, scale,
layer/head sharing, seed, LR, loss, batch, width/depth or duration. The next
mechanism must change a different scalable memory bottleneck rather than tune
this spectrum.
