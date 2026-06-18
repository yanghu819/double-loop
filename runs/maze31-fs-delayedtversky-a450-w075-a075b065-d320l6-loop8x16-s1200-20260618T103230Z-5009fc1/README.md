# Maze31 Delayed Tversky Probe

Recorded: 2026-06-18T11:05Z

## Hypothesis

Always-on Tversky pressure killed FutureSeed opening. This run keeps clean
FutureSeed training until after the historical step400 opening point, then
enables the same generic soft TP/FP/FN pressure at step450.

The point is not a weight sweep. It asks whether the previous failure was only
an opening/timing problem, or whether loss-only pruning still cannot create
late-loop self-correction.

## Config

- Remote row: GPU1 only, A800 80GB, `CUDA_VISIBLE_DEVICES=0`
- Git SHA: `5009fc1ff8d18c165f216652074f249467277d79`
- Task: Maze31 perfect mazes, path length `160-260`
- Model: FutureSeed+loop, hidden `320`, layers `6`, heads `8`
- State update: `none`
- Train/eval loops: `8 / 16`
- Steps/batch/eval: `1200 / 16 / 512`
- Tversky: `weight=0.75`, `after_step=450`, `start_loop=4`,
  `alpha=0.75`, `beta=0.65`

## Result

Opening was preserved, unlike the always-on Tversky run.

| step | train path F1 | active Tversky weight |
|---:|---:|---:|
| 100 | 0.0000 | 0.00 |
| 200 | 0.0000 | 0.00 |
| 300 | 0.0000 | 0.00 |
| 400 | 0.5455 | 0.00 |
| 500 | 0.5601 | 0.75 |
| 600 | 0.5566 | 0.75 |
| 900 | 0.5885 | 0.75 |
| 1200 | 0.5452 | 0.75 |

Held-out eval:

| loop | path F1 | precision | recall | pred PATH frac |
|---|---:|---:|---:|---:|
| loop1 | 0.5822 | 0.4305 | 0.9096 | 0.4082 |
| loop16 | 0.5824 | 0.4308 | 0.9091 | 0.4077 |

Visual hard-case average:

| loop | FP | FN |
|---|---:|---:|
| loop1 | 260.5 | 30.8 |
| loop16 | 260.2 | 31.3 |

## Decision

Delayed pressure is useful as training timing: it avoids the no-PATH attractor
and raises final loop16 path F1 from the clean D320/L6 run's `0.5344` to
`0.5824`.

It still does not prove the loop mechanism. Loop16 is nearly identical to loop1:
F1 gain is only `+0.0002`, false positives barely decrease, and false negatives
slightly increase. The improvement is a better overall operating point after
training, not recurrent self-correction.

Next work should not sweep Tversky weight or timing. The higher-ROI direction is
a simple generic recurrent state/decision dynamic that makes each later loop
receive a new input or state to revise, while keeping this delayed schedule as a
training guard if correction pressure is used.

## Artifacts

- `output/eqr_maze_probe_fs1_seed52.json`
- `output/visualizations/inline/summary_loop_effect.png`
- `output/visualizations/inline/case_best_loop_gain.png`
- `output/visualizations/inline/case_worst_loop_regression.png`
- `output/visualizations/inline/case_typical_failure.png`
- `logs/run.log`
- `config.json`, `score.json`, `metadata.json`
- `source_HEAD.txt`, `source.patch`, `source_snapshot.ls.txt`
