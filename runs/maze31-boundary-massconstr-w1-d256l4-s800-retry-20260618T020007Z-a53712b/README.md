# Maze31 Constrained Path-Mass Probe

Run: `maze31-boundary-massconstr-w1-d256l4-s800-retry-20260618T020007Z-a53712b`

## Hypothesis

The guarded path-mass objective preserved recall by disabling pruning gradients whenever the recall floor was violated, which froze the broad loop1 mask. This probe keeps the two pressures simultaneous: every active late loop still minimizes non-path PATH probability, while every true-path cell gets a loop1 probability floor.

The intended mechanism is generic: FutureSeed gives a broad direction, loop gives repeated computation, and training pressure teaches later loops to revise the state. There is no maze rule, search, repair, selector, seed sweep, or threshold sweep.

## Configuration

- Source SHA: `a53712b11835dd2d515fe9fff274a3e6a1d70532`
- GPU: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Task: Maze31 perfect mazes, path length `160-260`
- Model: `state_compete_boundary`, hidden `256`, layers `4`, heads `8`
- Training: `800` steps, batch `16`, train loops `6`, eval loops `12`
- Objective: `EQR_PATH_MASS_WEIGHT=1.0`, `EQR_PATH_MASS_START_LOOP=4`, `EQR_PATH_MASS_MODE=constrained`

The first launch with the same config halted at step200 because the GPU1 environment expired. That partial run is archived separately as `maze31-boundary-massconstr-w1-d256l4-s800-20260618T014930Z-a53712b`.

## Results

| metric | loop1 | loop12 | delta |
|---|---:|---:|---:|
| path F1 | 0.5836 | 0.5841 | +0.0005 |
| precision | 0.4144 | 0.4151 | +0.0007 |
| recall | 0.9984 | 0.9971 | -0.0013 |
| predicted PATH frac | 0.4655 | 0.4641 | -0.0014 |
| non-PATH PATH prob | 0.2556 | 0.2581 | +0.0025 |
| true-PATH PATH prob | 0.7642 | 0.7725 | +0.0082 |
| excess PATH prob frac | 0.1611 | 0.1647 | +0.0036 |
| positive floor loss | 0.0000 | 0.0035 | +0.0035 |
| positive floor violation frac | 0.0000 | 0.3071 | +0.3071 |

Hard visualized cases:

| hard-case average | loop1 | loop12 |
|---|---:|---:|
| false positives | 288.1 | 287.5 |
| false negatives | 0.5 | 1.3 |
| path F1 | 0.5247 | 0.5233 |
| precision | 0.3561 | 0.3554 |
| recall | 0.9969 | 0.9919 |

## Decision

This is not a useful positive result. The constrained objective avoids the two previous extremes: it does not freeze perfectly like `guarded`, and it does not collapse recall as much as `soft` mass pressure. But it also does not produce meaningful loop-time correction. The hard mask moves by only `-0.0014` predicted PATH fraction, hard-case false positives drop by only `0.6`, and false negatives still rise.

The key diagnostic is that the positive floor is active on `30.7%` of true-path cells at loop12, but the soft non-path mass does not improve; it slightly worsens. The objective is mostly trading tiny calibration changes rather than teaching a robust prune/keep decision.

Do not sweep this exact cell-floor probability objective. The next higher-ROI direction should change the recurrent decision variable, not just add another probability regularizer: for example a learned per-loop budget/normalization state, or a contrastive boundary objective that directly separates false-positive PATH candidates from true-path candidates while preserving true-path margins.

## Artifacts

- Metrics: `output/eqr_maze_probe_fs1_seed52.json`
- Report: `output/eqr_maze_probe_fs1_seed52.md`
- Visual casebook: `output/visualizations/index.html`
- Cases JSON: `output/visualizations/cases.json`
- Logs: `logs/launch.log`, `logs/run.log`
