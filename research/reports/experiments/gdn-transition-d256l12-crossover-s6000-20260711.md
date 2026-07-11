# D256/L12 Long-Training Crossover Test

## 1. Metainfo

- Plan ID: `P-SCALE-028`
- Status: discarded
- Planned: 2026-07-11 16:26 CST / 2026-07-11T08:26:05Z
- Launched: 2026-07-11 16:46 CST / 2026-07-11T08:46:00Z
- Completed: 2026-07-11 22:07 CST / 2026-07-11T14:07:28Z
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Tracking-plan SHA: `75fe60cf92baef66546067ed8c194692df7c2b4e`
- Formal source SHA: `ae31c9f54ffb0abaecf48e2102942431f57d36b3`
- Run name: `gdn-transition-d256l12-crossover-s6000-20260711T0846Z-ae31c9f`
- Resume run: `gdn-transition-d256l12-crossover-resume4900-s6000-20260711T1240Z-ae31c9f`

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

The first AIStation lease was proactively stopped after the complete step4900
checkpoint. Exact PIDs `697/695/696/652` were terminated, GPU1 returned to
`0 MiB`, and `lease_rollover_1.json` recorded the event. After GPU1 was restarted
and re-probed, `pscale028_resume4900.sh` resumed the exact optimizer/RNG/model
state from step4900 under outer PID `94`. It finished step6000 and all final
evaluations without a configuration change.

## 6. Artifacts

- Remote run directory:
  `/huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z/runs/gdn-transition-d256l12-crossover-s6000-20260711T0846Z-ae31c9f`
- Remote resume/final run directory:
  `/huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z/runs/gdn-transition-d256l12-crossover-resume4900-s6000-20260711T1240Z-ae31c9f`
- Train checkpoints:
  `/huyang2/double-loop/models/gdn-transition-d256l12-crossover-s6000-20260711T0846Z-ae31c9f/checkpoints`
- Launch script/log/PID/environment:
  `/huyang2/double-loop/artifacts/launch/pscale028/`
- Outer launch PID: `652`
- Resume outer PID: `94`
- Metadata archive:
  `/huyang2/double-loop/artifacts/gdn-transition-d256l12-crossover-s6000-20260711-ae31c9f-metadata-light.tgz`
- Archive SHA256: `7516d918c86dc7ef2dfecddb73db490806bd58fcd60d9de7c137dd8c45c58c93`
- Git-tracked primary run:
  `runs/gdn-transition-d256l12-crossover-resume4900-s6000-20260711T1240Z-ae31c9f`
- Human-readable comparison and hardest-case dashboard:
  `runs/gdn-transition-d256l12-crossover-resume4900-s6000-20260711T1240Z-ae31c9f/index.html`

## 7. Results

Interim checkpoint readout:

| Step | holes53 loop5 exact / blank | holes60 loop5 exact / blank | holes64 loop5 exact / blank |
|---:|---:|---:|---:|
| 1000, original P-DIAG-015 | `0.0176 / 0.5150` | `0.0215 / 0.5308` | not recorded |
| 2000 | `0.0195 / 0.5403` | `0.0254 / 0.5509` | `0.0371 / 0.5390` |
| 3000 | `0.0391 / 0.5530` | `0.0430 / 0.5661` | `0.0508 / 0.5532` |
| 4500 | `0.0566 / 0.5801` | `0.0664 / 0.5958` | `0.0723 / 0.5822` |
| 6000 | `0.0820 / 0.6156` | `0.1367 / 0.6408` | `0.1230 / 0.6196` |

At step3000, holes53 exact is slightly above the D224 matched-step reference
`0.0352`, while blank accuracy is still below D224 `0.5750`. The exact curve is
not flat: holes53 doubles from `0.0195` at step2000 to `0.0391` at step3000.
Harder buckets also retain genuine loop gains at step3000:

- holes60 exact: `0.0195 -> 0.0430`, loop1 to loop5.
- holes64 exact: `0.0234 -> 0.0508`, loop1 to loop5.

Decision at the predeclared step3000 gate: continue the unchanged run to
step4500/6000. This is only a weak crossover signal, not a success claim; the
primary step6000 and official-range gates remain unchanged.

Final mixed loop curve:

| Loop | Exact | Blank accuracy |
|---:|---:|---:|
| 1 | `0.0234` | `0.5383` |
| 2 | `0.0273` | `0.6021` |
| 3 | `0.0938` | `0.6272` |
| 4 | `0.1328` | `0.6323` |
| 5 | `0.1367` | `0.6327` |

Official blank-range loop5 readout:

| Range | Exact | Blank accuracy | D224 step6000 exact reference |
|---|---:|---:|---:|
| 46-50 | `1.0000` | `1.0000` | `1.0000` |
| 51-55 | `0.1504` | `0.6956` | `0.2832` |
| 56-64 | `0.0781` | `0.5834` | `0.0801` |

All three primary success gates fail:

- holes53 step6000 exact is `0.0820`, not greater than `0.1680`.
- official 51-55 exact is `0.1504`, below `0.30`.
- official 56-64 exact is `0.0781`, below `0.10`.

The sharpest diagnostic is the matched holes53 comparison. D256 has slightly
higher blank accuracy than D224 at step6000 (`0.6156` vs `0.6120`) but about
half the full-board exact (`0.0820` vs `0.1680`). More width improves local
token quality without converting it into global consistency as efficiently.

## 8. Conclusions

Decision: discard ordinary width scaling under this compute/data regime.

D256 is trainable and loop computation remains valuable. The negative result
is not that scale or loops do nothing: holes53 exact rises from `0.0176` at
step1000 to `0.0820` at step6000, and final mixed exact rises `0.0234 -> 0.1367`
from loop1 to loop5. The failed claim is stronger: D256 does not overtake the
cheaper D224 frontier. The weak step3000 crossing disappears as D224's exact
curve accelerates later.

This supports the complete scaling-law interpretation rather than blind
parameter scaling. A larger model must receive proportionally more independent
data and optimization to reach its better asymptote. Under the available
budget, D256 consumes more FLOPs and wall time while learning global exactness
more slowly. Do not run a D288/D320 table or rescue this result with LR, seed,
batch, or loss sweeps.

The next scalable direction should measure a compute-optimal frontier: keep the
more efficient D224 backbone, increase genuinely independent training boards
and tokens, and compare quality against training FLOPs/wall time. A different
state representation is justified only if it increases generic recurrent
capacity per FLOP, not as a Sudoku-specific repair.

## 9. Submission Record

No tag. Primary full score is `0.1367`, the matched D224 frontier is stronger,
and the width claim is negative. Configs, scores, logs, source patches, launch
scripts, exact-PID lease rollover provenance, checkpoint metrics, case banks,
and HTML visualizations are archived. Model checkpoints remain only under
`/huyang2/double-loop/models` and are not tracked by Git.
