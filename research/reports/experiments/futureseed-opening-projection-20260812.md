# P-LOOP-002: FutureSeed Opening-Gradient Projection

## 1. Metainfo

- Status: discarded after one complete matched A100 science run
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
- Formal UUID: `GPU-05f3e3f9-3f6f-82e9-1107-6e86672e77b5`
- Persistent root: `/huyang2/double-loop`
- Parent checkpoint SHA256: `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Pinned FLA source SHA: `9c8e42e762fce087c27b673af4922795d9edb85e`

Formal execution requires an exact pushed SHA and clean detached worktree.
CPU model smoke, GPU2, fallback and concurrent GPU model/eval are forbidden.

## 5. Commands

```bash
CUDA_VISIBLE_DEVICES=0 \
EXPECTED_UUID=GPU-05f3e3f9-3f6f-82e9-1107-6e86672e77b5 \
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

## 8. Results And Decision

The first formal attempt on UUID
`GPU-0da20a4f-5e67-e47d-7aab-8c6efa2864ad` lost its platform lease during
the canonical arm after the last logged step3075. It produced no canonical
checkpoint or comparison and is recorded separately as a non-science resource
interruption. No metric from that attempt enters the decision.

The fresh matched run
`p-loop-002-matched-20260812T172259Z-15657cc` completed status0 from exact
pushed source `15657cc3e232d455b1823eaf392765d7b78c7e35` on one visible
A100-SXM4-80GB. The strict contract, step3001 probe, same-source canonical arm,
candidate arm, full official evaluation and same-board comparison all
completed sequentially.

The intervention is mechanically strong and numerically exact. It activates
on 57/100 steps, touches exactly 11 receiving tensors and 88 scalars, removes
on average `0.831642` of the conflicting opening component when active, and
has mean active relative correction `0.126163`. The minimum post-projection
opening/continuation dot is `-1.22e-20`; FS-vector and final-global norm maximum
relative errors are `4.57e-8/9.59e-8`. Thus a no-op implementation cannot
explain the outcome.

Quality nevertheless fails both registered routes. Hard51-64 macro loop5
exact changes `0.001302 -> 0`, and mixed loop5 exact remains `0.025391`.
Official loop5 blank deltas for 51-55/56-60/61-64 are
`-0.001895/-0.000996/-0.005311`. Same-board loop3-to5 wrong-cell correction
improves on 51-55 (`+0.132812` cells/board relative to control) but weakens on
56-60 and 61-64 (`-0.027344/-0.007812`). Candidate train CE is `0.858957`.

Systems gates pass: candidate/control training time is
`971.40/1023.02s` (`-5.05%` overhead), peak allocation is
`13217.57/13186.49 MiB` (`+0.236%`), and inference delta is zero. The quality
miss is therefore not a cost artifact.

Discard and close the opening-gradient projection family. P-LOOP-001 found a
real aggregate gradient conflict, but removing that component from the native
FS gate gradient is not the causal hard-closure bottleneck at this parent.
Do not rescue with projection scale, per-head/per-layer selection, Adam-moment
editing, loss weighting, seed, LR, batch, width, depth or duration. A subsequent
diagnostic may inspect receiver-state cotangents, but it must select a genuinely
different receiver-native content or loop-dynamics mechanism rather than
another scalar-gate surgery.

Artifact SHA256 values are:

- CUDA contract: `c2196dbbd8f349c641a1d0e65313f721933c3fa3b9784324763094d148e997c5`;
- canonical metrics/checkpoint: `d73c629f45962269a619291a6d76e5980c9802c0e97f840e122e75bf366e8151` / `51cef631ae0388d9f0d3086127f71189aefb6b9914144979ba381fd8b9139719`;
- candidate metrics/checkpoint: `d419a7197e71e1b00aa26a1297221e33b8928e7eb5cdcbccf8ca46d6e6f29ccb` / `7233dcd4bb2c5e0feeee5154f811965baa12c00810b79b593899ade45be7efca`.

## 9. Submission Record

Not applicable. The complete local archive is under
`research/reports/visualizations/futureseed-opening-projection-20260812/`.
