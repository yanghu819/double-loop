# Maze31 EqR Baseline D320/L6 Abort

Recorded: 2026-06-18T09:08Z

## Hypothesis

If FutureSeed+loop is a better mainline than an EqR-style recurrent baseline on
hard Maze31, then removing FutureSeed under the same capacity, data, loop
budget, and training budget should hurt the ability to open PATH prediction.

This run is the equal-compute baseline for the previous clean FutureSeed+loop
run. It is not a seed sweep or a table-filling ablation; it answers whether the
main claim has a real baseline gap.

## Config

- Remote row: GPU1 only, A800 80GB, `CUDA_VISIBLE_DEVICES=0`
- Git SHA: `2cdb9c5db8f6225f10db583ddb36cb170854eeeb`
- Task: Maze31 perfect mazes, path length `160-260`
- Model: EqR maze runner, hidden `320`, layers `6`, heads `8`
- State update: `none`
- FutureSeed: `EQR_FUTURE_SEED_SCALE=0`
- Train/eval loops: `8 / 16`
- Planned steps/batch/eval: `1200 / 16 / 512`
- Actual stop: exact PID stop after step600 because PATH stayed unopened

## Result

The baseline never opened.

| step | CE | train path F1 |
|---:|---:|---:|
| 100 | 1.0625 | 0.0000 |
| 200 | 1.0703 | 0.0000 |
| 300 | 1.0625 | 0.0000 |
| 400 | 1.0703 | 0.0000 |
| 500 | 1.0625 | 0.0000 |
| 600 | 1.0625 | 0.0000 |

The matched FutureSeed+loop run opened at step400 with train path F1 `0.5862`
and finished with held-out loop16 path F1 `0.5344`. This baseline was stopped at
step600 because the marginal information gain of continuing was low.

## Decision

This is positive evidence for FutureSeed as an opening/optimization mechanism
over the EqR baseline on hard Maze31. It is not enough to claim the whole
FutureSeed+loop paradigm is better than EqR, because the matched FutureSeed run
still failed the second criterion: loop16 mostly added coverage and did not
cleanly reduce false positives.

The next proof target should be generic loop self-correction: later loops should
reduce false positives while keeping false negatives flat or lower. Do not spend
more budget on this exact baseline.

## Artifacts

- `abort.json`
- `config.json`
- `logs/run.log`
- `logs/launch.outer.log`
- `hypothesis.md`
- `launch.env`
- `source_HEAD.txt`, `source.patch`, `source_snapshot.ls.txt`
