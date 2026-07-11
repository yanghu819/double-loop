# D256/L12 Long-Training Crossover Test

## 1. Metainfo

- Plan ID: `P-SCALE-028`
- Status: in-progress
- Planned: 2026-07-11 16:26 CST / 2026-07-11T08:26:05Z
- Launched: 2026-07-11 16:46 CST / 2026-07-11T08:46:00Z
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Tracking-plan SHA: `75fe60cf92baef66546067ed8c194692df7c2b4e`
- Formal source SHA: `ae31c9f54ffb0abaecf48e2102942431f57d36b3`
- Run name: `gdn-transition-d256l12-crossover-s6000-20260711T0846Z-ae31c9f`

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

The source is the existing clean detached worktree that produced the resumed
D256 checkpoint:

`/huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z`

Before launch, the worktree reported exact HEAD
`ae31c9f54ffb0abaecf48e2102942431f57d36b3` and zero tracked changes. The
checkpoint is:

`/huyang2/double-loop/models/gdn-transition-d256l12-width-noconv-s6000-20260702T1245Z-ae31c9f/checkpoints/train_state_step001000.pt`

The reviewed launch script is:

`/huyang2/double-loop/artifacts/launch/pscale028/pscale028_launch.sh`

It exports the configuration in section 3 with `CUDA_VISIBLE_DEVICES=0`,
`SMOKE_DONE=1`, `SKIP_SETUP=1`, and `SOURCE_SNAPSHOT_MODE=lean`, then executes
`./run.sh full`. It was started under `nohup` with exact outer PID `652`; output
is written to
`/huyang2/double-loop/artifacts/launch/pscale028/pscale028_launch.out`.

No CPU model smoke was run. GPU-only preflight confirmed PyTorch `2.7.0+cu126`,
CUDA available, and `NVIDIA A800-SXM4-80GB` at `0 MiB` before launch. The first
runtime check confirmed checkpoint resume at global step1000 and approximately
`51886 MiB` allocated on GPU1.

## 6. Artifacts

- Remote run directory:
  `/huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z/runs/gdn-transition-d256l12-crossover-s6000-20260711T0846Z-ae31c9f`
- Train checkpoints:
  `/huyang2/double-loop/models/gdn-transition-d256l12-crossover-s6000-20260711T0846Z-ae31c9f/checkpoints`
- Launch script/log/PID/environment:
  `/huyang2/double-loop/artifacts/launch/pscale028/`
- Outer launch PID: `652`

## 7. Results

Interim checkpoint readout:

| Step | holes53 loop5 exact / blank | holes60 loop5 exact / blank | holes64 loop5 exact / blank |
|---:|---:|---:|---:|
| 1000, original P-DIAG-015 | `0.0176 / 0.5150` | `0.0215 / 0.5308` | not recorded |
| 2000 | `0.0195 / 0.5403` | `0.0254 / 0.5509` | `0.0371 / 0.5390` |
| 3000 | `0.0391 / 0.5530` | `0.0430 / 0.5661` | `0.0508 / 0.5532` |

At step3000, holes53 exact is slightly above the D224 matched-step reference
`0.0352`, while blank accuracy is still below D224 `0.5750`. The exact curve is
not flat: holes53 doubles from `0.0195` at step2000 to `0.0391` at step3000.
Harder buckets also retain genuine loop gains at step3000:

- holes60 exact: `0.0195 -> 0.0430`, loop1 to loop5.
- holes64 exact: `0.0234 -> 0.0508`, loop1 to loop5.

Decision at the predeclared step3000 gate: continue the unchanged run to
step4500/6000. This is only a weak crossover signal, not a success claim; the
primary step6000 and official-range gates remain unchanged.

## 8. Conclusions

Pending.

## 9. Submission Record

No tag unless the final score is strong and the scaling conclusion is clean.
