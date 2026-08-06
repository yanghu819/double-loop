# P-FS3-002: Shared Producer Update Codec

## 1. Metainfo

- Status: approved implementation; no GPU run launched
- Date: 2026-08-07
- Branch: `codex/fs3-producer-codec-20260807`
- Benchmark: official/full-diversity hard 9x9 Sudoku
- Compute: AIStation task-mode GPU1 only
- Seed: 52 only
- Parent mechanism: position-QK GDN3 plus native FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Frozen matched control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Mechanism Question

P-FS3-001 showed that a live, bounded residual based on analytic orthogonality
does not improve hard exact. Earlier FutureSeed2 experiments also closed raw
state carry, static KxV masks, scalar content trust, and fixed longer-radius
readout. The remaining content question is whether the producer should learn a
compact message from what its recurrent state actually wrote, rather than pass
only the raw terminal state or emphasize a hand-chosen direction.

P-FS3-002 learns one shared producer update codec. For receiver layers2+, let
`T` be the producer terminal state and `I` its exact incoming seed. The codec:

1. normalizes `T` and the real producer update `T-I` by terminal-state RMS;
2. scores every K row from five generic row statistics and compresses the
   KxV update to one V-dimensional payload code by softmax pooling;
3. decodes a bounded KxV residual from local terminal/update values, the shared
   payload code, and their interaction;
4. adds that residual before the unchanged native unit normalization and head
   gate.

The row scorer and cell decoder are shared across every layer and head. Their
operations are equivariant to independent K-row and V-column permutations, so
the mechanism is not a learned layer-specific basis transport. The bottleneck
width is exactly the existing V state width; there is no rank or codec-width
knob. Only the final one-output decoder is zero initialized. The complete model
therefore starts bit-exact to terminal FutureSeed while retaining a gradient
path into a materially more expressive content transform.

This adds 59 parameters at D256/L12/H8/K32/V32. It adds no recurrent step,
scan, layer, Sudoku feature, rule, search, repair, selector, oracle state, or
reverse pass. Pinned official FLA GDN2/Triton recurrence and position-QK remain
unchanged.

## 3. Registered Experiment

Do not rerun the terminal control. Its exact step3000-to3100 continuation is
already frozen with the same parent checkpoint, optimizer, RNG, data order,
architecture, effective batch128, and seed52. After a pushed source SHA and a
clean detached worktree pass all contracts, run exactly one producer-codec arm
from the same parent to step3100. The candidate alone sets:

```text
future_seed_content_mode=producer_codec
resume_allow_future_seed_content_upgrade=true
```

Every other semantic argument must match the frozen control. The checkpoint
loader must report exactly the six producer-codec parameter tensors as missing,
no unexpected tensors, and deterministic optimizer-group expansion.

## 4. Preflight Gates

The single-GPU CUDA contract must pass before the formal continuation:

- only CUDA index0 and UUID
  `GPU-53e9f3b4-2966-65d3-6614-09c540921519` are visible;
- source is a clean detached GitHub SHA and the parent hash/source match;
- zero-init model output and every layer terminal state are bit-exact to
  terminal FutureSeed;
- parameter delta is exactly 59 and checkpoint migration is exactly the six
  registered tensors;
- the zero decoder gets a finite nonzero gradient, and after opening it every
  codec parameter gets finite nonzero gradient;
- independent row/column permutations commute with the FP32 codec to max error
  `<=2e-6`, and residual RMS stays bounded by terminal RMS;
- all 12 layers use the pinned official `GatedDeltaNet2`,
  `ChunkGDN2FunctionBackward`, and Triton convolution with position-QK active;
- no NaN, OOM, fallback, data drift, or second compute app occurs.

Any miss blocks the run. Fix an implementation defect only before formal
launch; do not change the mechanism or gate after seeing model quality.

## 5. Science Decision

Activation requires finite nonzero payload-code RMS, codec residual relative
RMS at least `1e-4`, and nonzero between-board residual variation. Report row
attention entropy/max/batch variation even if routing remains nearly uniform.

Primary quality pass:

- hard official51-64 macro loop5 exact improves by at least `+0.02` over the
  frozen terminal control; and
- no individual official blank range regresses by more than `0.01` blank
  accuracy.

Alternate quality pass:

- mixed loop5 exact improves by at least `+0.03`;
- official61-64 exact is non-regressive; and
- same-board loop3-5 wrong-cell correction is stronger across the hard ranges.

Cost requires candidate 100-step elapsed overhead below 10 percent versus the
frozen control and no material peak-allocation regression beyond 10 percent.
Report both totals and timing stability; quality cannot be rescued by a timing
caveat.

Kill and discard on dead activation, either quality-route miss, cost miss,
integrity failure, or broad blank-only movement without stronger exact/loop
closure. Do not rescue with scorer features, bottleneck width, rank, decoder
scale, seed, LR, loss, batch, model width/depth, or duration.

## 6. Required Artifacts

Archive the candidate config, score JSON, checkpoint, source snapshot, launch
and contract logs, all hashes, official51-55/56-60/61-64 and mixed loop1-5
metrics, CE, activation, throughput, peak memory, and same-board loop
visualization. Update `plans.md`, `leaderboard.csv`, `docs/PAPER_PLAN.md`, and
`docs/LESSONS.md`, then commit and push the decision before choosing a
successor.

## 7. Conclusion

Pending the strict CUDA contract and one registered candidate continuation.
Success would support learned shared producer-message formation as a stronger
FutureSeed interface. Failure closes this compact permutation-equivariant
codec, not all learned state communication, and does not authorize a codec
feature/rank/width sweep.
