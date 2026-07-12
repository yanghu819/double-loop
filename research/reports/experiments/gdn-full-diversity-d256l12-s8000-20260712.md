# Full-Diversity D256 Joint Scaling Test

## 1. Metainfo

- Plan ID: `P-SCALE-032`
- Status: in progress
- Planned: 2026-07-12 CST
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Experiment source SHA: resolved by the planning commit and recorded in `launch.env`
- Training mode: from scratch, resumable exact checkpoints

## 2. Mechanism Hypothesis

P-SCALE-028 showed that D256/L12 does not beat D224/L12 when both are trained on
only 1,000 source boards repeated through augmentations. P-SCALE-029 then showed
that 3.83M independent source boards sharply improve D224 after a delayed
crossover around step4500.

The unresolved scaling question is therefore joint rather than one-dimensional:
does the larger D256 backbone need independent relational data to use its extra
capacity? If yes, D256 should start no better than D224 but eventually overtake
the D224 full-diversity curve. If it does not, ordinary width is not an effective
capacity-per-compute axis for the current FutureSeed-GDN state formulation.

## 3. Intervention

- Train native FutureSeed GDN D256/L12/H16/D16 from scratch.
- Keep `GDN_EXPAND_V=4.0`, no short convolution, and fixed FutureSeed/update.
- Keep loop5 with equal CE supervision on every loop.
- Use BF16 Triton recurrent forward/backward and effective batch128.
- Use all 3,831,994 independent training boards.
- Match the successful D224 curriculum exactly:
  `46-50:100,51-55:5900,51-64:2000`.
- Evaluate holes53/60/64 at steps 1000, 3000, 4500, 6000, 7000, and 8000.
- Add no noise, scratch state, gate, repair, search, selector, or Sudoku rule.

## 4. Prediction, Budget, And Kill Criteria

Budget: approximately 6-7 GPU hours across resumable GPU1 leases. A single
GPU-only optimizer-step fit gate precedes the formal run; no CPU smoke is used.

Prediction and gates:

- do not quality-stop at step1000;
- at step4500, stop only if all hard-bucket exact values are below `0.07`, all
  blank accuracies are below `0.60`, and the curve is not accelerating;
- primary step6000: holes53 exact `>=0.21` and holes60 or holes64 `>=0.25`;
- continue to step8000 only if D256 reaches or clearly approaches the D224
  full-diversity step6000 curve;
- strong step8000: beat D224 in at least two fixed buckets and improve official
  51-55 `0.4473` or 56-64 `0.1621`.

Stop immediately on wrong GPU, source ambiguity, NaN, OOM, or malformed
checkpoint. Do not rescue a miss with width, seed, LR, loss, or schedule sweeps.

## 5. Paper Decision

Success supports joint data/model scaling: FutureSeed-GDN can use more generic
capacity when trained on enough independent relations. Failure establishes D224
as the more compute-efficient frontier and redirects work toward a more
efficient generic recurrent state, not more parameters or task-specific tricks.

## 6. Results

Pending.

## 7. Conclusions

Pending.

## 8. Artifacts And Visualization

Pending. Every scored checkpoint must archive fixed-bucket metrics; the final
decision checkpoint must include loops1/3/5 hard-case visualizations.

## 9. Submission Record

No tag unless score reaches `>=0.50` and the scaling claim is clean.
