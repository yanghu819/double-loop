# D256/L12 Long-Training Crossover Test

## 1. Metainfo

- Plan ID: `P-SCALE-028`
- Status: approved
- Planned start: 2026-07-11 16:26 CST / 2026-07-11T08:26:05Z
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Source SHA: assigned by the prelaunch tracking commit
- Run name: assigned after the source SHA is fixed

## 2. Hypothesis

The previous D256/L12 run was stopped at step1000 because it tied exact and
slightly trailed D224/L12 blank accuracy at the same step. That is not a fair
test of a scaling-law crossover: larger models often need more optimization
tokens before their better asymptote appears. D224/L12 itself did not show a
strong exact rise until after step3000.

This experiment asks one question: when D256/L12 receives enough broad hard
data and training time, does its exact curve become steeper than D224/L12, or
is ordinary width genuinely the wrong scaling axis for this recurrent model?

## 3. Configuration

- Resume checkpoint: P-DIAG-015 D256/L12 exact step1000 train checkpoint.
- Data: official EqR Sudoku arrays; random blank masks.
- Backbone: native FutureSeed GDN with Triton recurrent forward/backward.
- Model: D256/L12/H16/head-dim16, `GDN_EXPAND_V=4.0`, no short convolution.
- FutureSeed normalization: existing `unit` RMS behavior, not adaptive RMS.
- Recurrent compute: loop5, `L_CYCLES=2`, every-loop CE.
- Batch: microbatch32, gradient accumulation4, effective batch128.
- Global curriculum:
  - `46-50:100`
  - `51-55:900`
  - `51-64:1000`
  - `51-55:500`
  - `56-64:500`
  - `51-64:1000`
  - `51-55:500`
  - `56-64:500`
  - `51-64:1000`
- Target: step6000; checkpoint evaluations at 2000, 3000, 4500, 6000 on
  holes53/60/64.
- Final evaluation: official ranges 46-50, 51-55, 56-64 and case-bank loops
  1/3/5.
- Disabled: adaptive RMS, feature/aggregate noise, scratch, learned gate, extra
  loss, repair, search, selector, oracle rollout, task-specific rules, seed or
  learning-rate sweep.

## 4. Prediction And Kill Criteria

Prediction:

- By step3000, D256 should at least match D224 step3000 holes53 loop5
  `0.0352` exact / `0.5750` blank accuracy and show an accelerating exact
  slope.
- By step6000, holes53 loop5 exact should exceed D224's `0.1680`.
- Final official 51-55 should be at least `0.30`, and 56-64 at least `0.10`.
- Loop1-to-loop5 exact gain must remain substantial; a local blank-only gain is
  not a scaling success.

Kill criteria:

- Stop on wrong GPU, OOM, NaN, missing checkpoint provenance, or unexpected
  architecture mismatch.
- At step3000, stop by exact PID if holes53 remains below the D224 matched-step
  reference and the 2000-to-3000 exact slope is flat.
- Stop if throughput makes step6000 infeasible in the current GPU1 lease.
- Do not rescue failure with a seed, LR, batch, width, or loss table.

## 5. Commands

The exact detached worktree, checkpoint path, launch script, and command will be
recorded immediately after the prelaunch tracking SHA is pushed and GPU1 is
probed. No CPU model smoke is permitted.

## 6. Artifacts

Pending launch.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

No tag unless the final score is strong and the scaling conclusion is clean.
