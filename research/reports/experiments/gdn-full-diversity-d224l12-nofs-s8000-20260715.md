# Matched Full-Diversity No-FutureSeed Control

## 1. Metainfo

- Plan ID: `P-SCALE-033`
- Status: approved for launch
- Planned: 2026-07-15 CST
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Training mode: from scratch with exact resumable checkpoints

## 2. Mechanism Hypothesis

The strongest D224 result jointly uses FutureSeed, 3.83M independent boards,
five recurrent loops, and CE supervision at every loop. It proves that the full
system scales, but does not isolate which part supplies the opening advantage.

FutureSeed is intended to place a learned summary of future tokens into a
left-to-right recurrent state before the normal scan. If that mechanism is
useful, disabling only this state should make the otherwise matched GDN learn
hard boards later, require more recurrent compute, or fail to open. If the
no-FutureSeed model follows the same curve, FutureSeed cannot be the paper's
core causal claim.

## 3. Intervention

- Match D224/L12/H14/D16, GDN expand-v4, and BF16 Triton recurrent kernels.
- Match loop5 with equal CE supervision at every loop.
- Match effective batch128 and the 3,831,994 independent-board split.
- Match curriculum `46-50:100,51-55:5900,51-64:2000` through step8000.
- Change only `FUTURE_SEED_SCALE=1` to `0`; keep the parameter path present.
- Evaluate holes53/60/64 at steps 1000, 3000, 4500, 6000, 7000, and 8000.
- Report loop1-to-loop5 exact/blank, wall time, and peak GPU memory.
- Add no noise, scratch, repair, search, selector, or Sudoku-specific rule.

## 4. Prediction, Budget, And Kill Criteria

This is one seed and one arm, not an ablation table. The maximum budget is about
five GPU hours across exact-checkpoint leases. The first lease runs only as far
as can be safely checkpointed before platform shutdown; no CPU smoke is used.

The FutureSeed claim is supported if the matched no-FS curve is materially
slower or weaker, with a hard exact gap of at least `0.03`, or if no-FS needs at
least `20%` more compute to reach the same score. If no-FS reaches the same
exact, blank accuracy, and speed, the core claim is rejected.

Do not stop at step1000 because full-diversity learning is delayed. Stop at the
predeclared step4500 gate only if holes53/60/64 loop5 exact are all below `0.02`
and blank accuracies are all below `0.55`. Stop immediately on wrong GPU, dirty
source, malformed checkpoint, NaN, or OOM. Do not rescue a miss with seed, LR,
loss, width, schedule, or noise sweeps.

## 5. Paper Decision

- no-FS stays closed: FutureSeed supplies a necessary cheap future-context
  initialization under this compute budget;
- no-FS opens later but reaches parity: FutureSeed is an optimization and
  compute-efficiency mechanism;
- no-FS matches the curve: FutureSeed is not the main cause, and the paper must
  pivot to data plus recurrent supervision rather than claim future context.

## 6. Results

Pending.

## 7. Artifacts And Visualization

Every scored checkpoint will archive config, score, logs, source SHA, fixed
evaluation, and final loops1/3/5 case-bank visualization. Intermediate lease
rollovers archive an exact checkpoint and `lease_rollover.json`.

## 8. Submission Record

No tag unless the mechanism conclusion is strong and the primary score is at
least `0.50`.
