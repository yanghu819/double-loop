# Maze31 Clean D320/L6 Scale Probe Abort

Recorded: 2026-06-18T05:51:36Z

## Hypothesis

The preceding `state_compete_cross` scale run stayed at path F1 `0.0` through
step400. This short fallback removed that complex generic state update and kept
only clean FutureSeed + loop at the same D320/L6/train8/eval16 scale.

This was not a table-filling ablation. It answered one decision: was the
no-PATH collapse caused by the harder Maze31 scale schedule itself, or by the
more complex cross-candidate state dynamics?

## Config

- GPU row: GPU1 only, A100 80GB, `CUDA_VISIBLE_DEVICES=0`
- Source SHA: `8df4df222b946363e5c854b4e6765e7efa851eb1`
- Task: 31x31 perfect mazes, path length 160-260
- Model: hidden 320, layers 6, heads 8, 8.57M parameters
- Train/eval loops: 8 / 16
- Planned steps: 400
- Batch/eval: 16 / 512
- FutureSeed scale: 1
- State update: `none`
- Path loss weight: 1.5
- Extra pruning/correction losses: all disabled

## Observed Curve

| step | CE | path F1 | exact |
|---:|---:|---:|---:|
| 100 | 1.0625 | 0.0000 | 0.0000 |
| 200 | 1.0703 | 0.0000 | 0.0000 |
| 300 | 1.0625 | 0.0000 | 0.0000 |
| 400 | 0.3691 | 0.5861 | 0.0000 |

The process was killed just after the step400 training log, before final eval
and visualizations were written. This run therefore has no loop16 held-out
score, but the training curve is informative.

## Decision

The no-PATH failure is not an unavoidable consequence of D320/L6/loop8 scale.
Clean FutureSeed + loop eventually opened PATH prediction by step400. The
matched `state_compete_cross` scaled run did not.

The next useful run should give the clean D320/L6 setup enough budget to finish
evaluation and inspect loop1 -> loop16 pruning. Do not expand the complex
cross-candidate state update until the clean scale readout is known.

## Artifacts

- `abort.json`
- `config.json`, `metadata.json`, `score.json`
- `logs/run.log`, `logs/launch.log`
- `source_HEAD.txt`, `source.patch`
