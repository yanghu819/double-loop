# P-FS2-013: Loop-Secant FutureSeed

## 1. Metainfo

- Status: approved, implementation corrected; strict CUDA contract pending
- Date: 2026-08-15
- Branch: `codex/fs2-loop-secant-20260815`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: one AIStation A100 80GB, CUDA index 0 only
- GPU UUID: `GPU-d2877fe4-641c-fe64-2a74-8abca47c292f`
- Seed: 52 only
- Parent: D256/L12/H8/K32/V32 position-QK GDN3 plus native terminal
  FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen control: `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Evidence Boundary

The current evidence separates two failures. Directional MQAR localizes the
dominant GDN2 failure to value evidence being committed to the adjacent
same-direction owner; exact dual-key reconstruction then shows that changing a
trained address basis after the fact destroys the jointly learned query,
erase, write and read closure. Sudoku shows a different endpoint symptom:
native FutureSeed often makes genuine corrections in loops 2-3 but then
plateaus or oscillates rather than closing the board.

Prior loop mechanisms do not test a directional derivative of the transported
recurrent state. `loop_residual` interpolated the previous and current seed;
learned update gates changed only interpolation rate; logit feedback, gated
scratch and extra loops did not create late exact closure. P-FS3-001 changed
within-edge terminal content using an innovation residual, not the direction
of change across repeated reasoning passes.

## 3. Mechanism

For each adjacent producer-to-receiver edge `e` and head `h`, retain the
producer terminal state from the preceding application of the same reasoning
stream. Let the current and preceding producer terminals be `T_r` and
`T_{r-1}`. Form the secant

`D_r = T_r - T_{r-1}`.

Bound its RMS to the current terminal RMS and extrapolate from the current
state:

`S_seed = T_r + 0.5*tanh(a_eh)*bounded(D_r)`.

The first pass has no history and remains native. Parameters `a_eh` are zero
initialized, so every pass is exactly the parent function at launch. On the
high stream, consecutive applications are macro loops; on the low stream they
also include consecutive L-cycle passes. This deliberately tests whether the
direction in which the recurrent state is already moving is useful future
evidence, without adding tokens, scans, caches, new recurrent state, Sudoku
logic or a reverse pass.

For D256/L12/H8, the exact parameter delta is `11*8 = 88`; persistent model
state, physical GDN2 state, scan count and official FLA kernels are unchanged.

## 4. Falsifiable Prediction

If the remaining Sudoku failure is loop convergence rather than missing
address capacity, all 88 coefficients should receive gradient immediately and
the learned secant should show nonzero board variation. A 100-step exact
continuation should improve hard-board exact closure or mixed exactness while
preserving the hardest range and strengthening same-board loop3-to-loop5
correction. If the path activates but exact remains flat or correction weakens,
cross-pass state direction is not sufficient and this family is closed. If the
strict identity, official-kernel or RMS bound fails, the implementation claim
is false before science scoring.

## 5. Strict CUDA Contract

Exact pushed source in a clean detached worktree must prove all of:

1. only CUDA index 0 and UUID
   `GPU-d2877fe4-641c-fe64-2a74-8abca47c292f` are visible;
2. pinned FLA SHA is
   `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. all 12 layers are official `GatedDeltaNet2`, with
   `ChunkGDN2FunctionBackward` in every layer graph and no fallback;
4. zero initialization preserves full output and all 12 terminal states
   bit-exactly across five varying passes;
5. arbitrary finite nonzero prior-pass producer memories also preserve output
   and all states bit-exactly at zero initialization;
6. returned producer memory equals the live producer terminal state exactly;
7. parameter delta is exactly 88 and all 88 coefficients receive finite,
   nonzero direct gradients;
8. an opened path changes output, has nonzero state delta and board variation,
   remains finite, and satisfies coefficient and residual RMS bounds `<=0.5`;
9. no CPU model path, concurrent GPU process, NaN, OOM, source/data drift or
   silent fallback occurs.

## 6. Step3001 Production Gate

The one-step exact-resume probe is migration and production-fit evidence only,
not a quality measurement. It must:

- accept only the explicit `fixed -> loop_secant` semantic upgrade;
- add exactly `reasoner.future_seed_secant_raw` to optimizer groups while
  preserving optimizer/RNG/data order;
- complete status 0 with source/config/checkpoint/metric provenance;
- report finite nonzero secant scale, residual relative RMS and board variation
  after one update;
- retain the target GPU, official FLA graph and clean source contract.

## 7. Matched Science Gate

Only after Sections 5-6 pass, run one candidate-only step3000-to-step3100
continuation. Do not repeat the frozen control.

- Primary quality route: hard51-64 macro loop5 exact improves by `>=0.02`
  over control and each official hard range blank accuracy regresses by no more
  than `0.01`.
- Alternate quality route: mixed loop5 exact improves by `>=0.03`, official
  61-64 does not regress, and same-board loop3-to-loop5 wrong-cell correction
  is stronger.
- Activation: mean absolute secant coefficient `>=1e-4`; delta/residual RMS and
  board variation finite and nonzero.
- Cost: independent warmed elapsed overhead `<10%` and peak allocation
  overhead `<10%` versus the frozen control protocol.

Any integrity, activation, quality or cost miss discards P-FS2-013. There is no
coefficient cap, initialization, edge subset, seed, LR, loss, batch, width,
depth or duration rescue.

## 8. Required Endpoint Evidence

Report official 51-55/56-60/61-64 and mixed loop1-5 exact, blank accuracy and
wrong-cell counts; train CE; secant coefficient, delta and residual metrics;
same-board loop trace; independent warmed throughput; peak allocated/reserved
memory; timing stability; source/config/metric/checkpoint hashes.

## 9. Decision

Pending strict CUDA contract. Contract R1 source `56274f50` stopped before a
science run at its first dual-model identity comparison. The checker had not
reset the construction RNG independently for control and candidate, unlike the
established matched-contract harnesses. R2 changes only that harness setup and
adds the measured max error to any repeated failure; mechanism, prediction and
all gates remain frozen. R2 source `bee012a9` still found a large pass1 output
gap, so R3 additionally proves exact shared state-dict migration, fixed-control
self-repeatability and per-layer terminal-state errors before classifying the
failure. This remains harness localization; no training or gate changed.
R3 source `fa2f70aa` proved that shared state-dict migration and repeated
fixed-mode execution are exact; the two independently constructed FLA module
instances diverged beginning at layer 1 even before any secant history existed.
R4 therefore compares `fixed` and `loop_secant` semantics on the same migrated
candidate instance while retaining the independent model for exact state-dict
and parameter-delta checks. This directly isolates the new branch and avoids
misclassifying independent-instance runtime state as mechanism behavior. R4
then localized a real implementation defect: the branch computed
`candidate_seed_state` correctly but the common tail immediately overwrote it
with an unset `seed_state`, disabling FutureSeed rather than preserving the
zero-coefficient parent. R5 assigns the computed candidate into the common
seed variable before that tail. This is a semantic correctness fix discovered
before any training; the mechanism, initialization, parent, budget and all
registered gates remain unchanged.
