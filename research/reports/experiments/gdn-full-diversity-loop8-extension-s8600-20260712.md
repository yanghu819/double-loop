# Full-Diversity Loop-8 Compute Extension

## 1. Metainfo

- Plan ID: `P-SCALE-031`
- Status: in progress
- Planned: 2026-07-12 CST
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Experiment source SHA: resolved by the planning commit and recorded in `launch.env`
- Resume checkpoint: P-SCALE-030 exact step8000

## 2. Mechanism Hypothesis

P-SCALE-030 improved loop4/5 while loop1 stayed unchanged. That is direct evidence
that the current model learns corrections in later recurrent passes. The next
high-information question is whether the correction process is still limited by
the number of supervised passes, or whether the recurrent state has already
exhausted the useful information it can carry by loop5.

If recurrent compute is still the limiting resource, extending the same shared
block from five to eight supervised loops should continue reducing hard-board
errors. If loop6-8 merely copy loop5, then adding loops is not a useful scaling
axis and the next budget must move to a larger full-diversity backbone.

## 3. Intervention

- Resume the exact step8000 model, optimizer, scaler, feature buffer, and RNG.
- Keep native FutureSeed GDN D224/L12/H14/D16 and `GDN_EXPAND_V=4.0`.
- Keep the 3,831,994 independent-board dataset and 51-64 hard exposure.
- Change only `MAX_LOOPS=5` to `MAX_LOOPS=8`.
- Keep `LOOP_LOSS=all`, so every one of the eight loops receives CE supervision.
- Train 600 additional steps, with eval at steps 8200, 8400, and 8600.
- Do not add noise, scratch state, repair, search, selector, or task rules.

## 4. Prediction, Budget, And Kill Criteria

Prediction:

- loop6-8 should improve holes60 or holes64 exact by at least `+0.02` over loop5;
- official 56-64 exact should reach at least `0.18` while 51-55 remains `>=0.42`;
- hardest-case wrong-cell counts should continue falling after loop5.

Budget: one GPU-only fit step, then at most 600 resumed training steps on GPU1,
estimated 0.9-1.2 GPU hours plus evaluation.

Kill:

- stop immediately on wrong GPU, source/checkpoint mismatch, NaN, or OOM;
- at step8200, stop if both holes60 and holes64 have loop8-loop5 exact below
  `+0.01` and the visual cases show no error reduction after loop5;
- do not respond to failure by sweeping loop count, loop-loss weights, seed, LR,
  or schedule.

## 5. Paper Decision

Success supports the claim that loop count is a useful variable-compute axis:
the same parameters can spend more recurrent computation and solve more boards.
Failure establishes a clean compute ceiling at loop5 and redirects scaling to
larger generic state/model capacity. It does not by itself establish a
FutureSeed advantage over matched no-FutureSeed or EqR.

## 6. Results

Pending.

## 7. Conclusions

Pending.

## 8. Artifacts And Visualization

Pending. Final visualization must include input, target, and loops 1/3/5/6/7/8
for solved-by-loop, almost-solved, and hard-failure cases.

## 9. Submission Record

No tag unless the primary project score reaches `>=0.50` and the mechanism claim
is clean.
