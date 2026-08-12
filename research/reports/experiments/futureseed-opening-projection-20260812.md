# P-LOOP-002: FutureSeed Opening-Gradient Projection

## 1. Metainfo

- Status: approved; source preparation before launch
- Parent: position-QK GDN2 plus native terminal FutureSeed, exact step3000
- Task: official/full-diversity hard 9x9 Sudoku, 51-64 blanks
- Candidate: training-only FutureSeed opening-gradient projection

## 2. Hypothesis

P-LOOP-001 found a narrow optimization conflict rather than a global GDN
failure. On 51-55 and 61-64 boards, loop1 FutureSeed-gate gradients oppose
loops3-5, while loops3-5 are nearly collinear and the model genuinely corrects
cells over loops. Removing only the loop1 component that points against the
continuation direction may retain FutureSeed opening while improving late
closure.

This does not change GDN2 recurrence, FutureSeed forward transport, losses,
parameters, optimizer state, logits, inference cost or Sudoku logic. GDN2
address/edit gradients were aligned in P-LOOP-001, so no GDN arm is authorized.

## 3. Configuration

Both formal arms exact-resume the same SHA-locked D256/L12/H8/K32/V32 step3000
checkpoint to step3100 with five equally supervised loops, effective batch128,
BF16, seed52, random traversal, position-QK, native terminal FutureSeed and
pinned official FLA GDN2/Triton.

Let `O` be the effective-batch loop1 gradient on all receiving
`future_seed_logit` scalars and `B` the canonical equal-five-loop gradient.
Recover `C=(5B-O)/4`. If `O dot C<0`, set
`O'=O-(O dot C)/(||C||^2) C`, recombine `(O'+4C)/5`, and rescale only this
88-scalar FS subvector to the canonical FS gradient norm. Canonical global
clipping occurs first; `O` receives the same canonical clip coefficient. No
second global clipping is allowed, so every non-FS optimizer input remains
bitwise identical to control.

The matched sequence is strict CUDA contract, candidate-only step3001 probe,
same-source canonical step3000-to3100 control, then candidate. It is strictly
sequential with no concurrent model/eval.

## 4. Environment

- AIStation development row: `GPU2`
- Resource: one A100-SXM4-80GB
- Required CUDA view: index0 only
- Required UUID: `GPU-0da20a4f-5e67-e47d-7aab-8c6efa2864ad`
- Persistent root: `/huyang2/double-loop`
- Parent checkpoint SHA256: `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Pinned FLA source SHA: `9c8e42e762fce087c27b673af4922795d9edb85e`

Formal execution requires an exact pushed SHA and clean detached worktree.
CPU model smoke, GPU2, fallback and concurrent GPU model/eval are forbidden.

## 5. Commands

```bash
CUDA_VISIBLE_DEVICES=0 \
EXPECTED_UUID=GPU-0da20a4f-5e67-e47d-7aab-8c6efa2864ad \
PUSHED_REF=refs/heads/codex/futureseed-opening-projection-20260812 \
scripts/run_futureseed_opening_projection_matched.sh
```

## 6. Artifacts

The wrapper writes the CUDA contract below
`/huyang2/double-loop/artifacts/p-loop-002/<source-sha>/`, immutable probe and
formal run directories below `/huyang2/double-loop/runs/`, step3001/3100
checkpoints below `/huyang2/double-loop/models/`, and a final matched
`decision.json` with source, metrics, checkpoint and contract hashes.

## 7. Frozen Gates

Integrity and activation require exactly 12 gate tensors/96 scalars, exactly 11
receiving tensors/88 active scalars, block0 `.grad=None`, 12 official
`ChunkGDN2FunctionBackward` paths, exact forward/state/parameter identity,
continuation recovery error within the CUDA contract, activation on at least
10% of the 100 steps, mean active removed-opening fraction at least 5%,
post-projection dot at least `-1e-6`, FS norm relative error below `1e-5`, and
final global norm relative error below `2e-5`.

Primary quality requires hard51-64 macro loop5 exact at least `+0.02` over the
same-source control and each official range's blank regression no worse than
`0.01`. Alternate quality requires mixed exact `+0.03`, non-regressive 61-64
exact, 61-64 blank regression no worse than `0.005`, same-board 61-64 loop3-to5
wrong-cell correction at least `+0.5` cells/board, and non-weaker 56-60 late
correction. Elapsed overhead must be below 25% and peak allocation overhead
below 10%; inference overhead is exactly zero.

## 8. Decision

Pending. Any contract, production, activation, quality or cost miss closes this
mechanism. No loss weighting, projection scale, per-head variant, seed, LR,
batch, width, depth or duration rescue is permitted. A pass establishes a
better FutureSeed training rule, not a new GDN3 recurrence.

## 9. Submission Record

Not applicable.
