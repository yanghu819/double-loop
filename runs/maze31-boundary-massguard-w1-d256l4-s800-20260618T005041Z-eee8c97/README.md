# Maze31 Guarded Path-Mass Probe

Run: `maze31-boundary-massguard-w1-d256l4-s800-20260618T005041Z-eee8c97`

## Hypothesis

The previous late-loop path-mass pressure proved that recurrent loops can be pushed from high-recall expansion toward pruning, but it cut true-path cells as well as false positives. This probe tests a stricter generic recall guard: when a later loop's true-path probability drops below the loop1 floor, train only the guard; otherwise train non-path PATH-mass reduction.

This stays within the intended mechanism: FutureSeed proposes a broad direction, loop supplies repeated computation, and training pressure/state dynamics decide whether later loops can revise the broad mask. It does not encode maze rules, search, repair, selector logic, or a seed/weight sweep.

## Configuration

- Source SHA: `eee8c97a43c62ff508f365a2e573e8d10bf72d1d`
- GPU: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Task: Maze31 perfect mazes, path length `160-260`
- Model: `state_compete_boundary`, hidden `256`, layers `4`, heads `8`
- Training: `800` steps, batch `16`, train loops `6`, eval loops `12`
- Objective: `EQR_PATH_MASS_WEIGHT=1.0`, `EQR_PATH_MASS_START_LOOP=4`, `EQR_PATH_MASS_MODE=guarded`

## Results

| metric | loop1 | loop12 | delta |
|---|---:|---:|---:|
| path F1 | 0.5847 | 0.5847 | +0.0000 |
| precision | 0.4152 | 0.4152 | +0.0000 |
| recall | 1.0000 | 1.0000 | +0.0000 |
| predicted PATH frac | 0.4653 | 0.4653 | +0.0000 |
| non-PATH PATH prob | 0.2627 | 0.2561 | -0.0065 |
| true-PATH PATH prob | 0.7882 | 0.7693 | -0.0190 |
| excess PATH prob frac | 0.1714 | 0.1625 | -0.0090 |

Hard visualized cases also froze:

| hard-case average | loop1 | loop12 |
|---|---:|---:|
| false positives | 288.2 | 288.2 |
| false negatives | 0.0 | 0.0 |
| path F1 | 0.5246 | 0.5246 |

The guard was active on all eval samples at loop12: `path_mass_guard_violation_frac=1.0`. That explains the result. The objective preserved recall, but when the guard activates it removes the non-path mass gradient, so the model has no pressure to prune false positives at the hard decision boundary.

## Decision

Do not sweep `path_mass_weight`, margin, seed, or threshold around this guarded form. The useful conclusion is structural: recall-preserving pruning cannot be implemented by simply stopping mass pressure whenever true-path confidence falls below a floor. That creates a copy-loop failure mode.

The next higher-ROI direction is an objective that keeps positive-cell margins while still applying a dense false-positive penalty, for example a normalized margin objective or constrained/Lagrangian form where non-path mass and positive-margin preservation remain simultaneous signals.

## Artifacts

- Metrics: `output/eqr_maze_probe_fs1_seed52.json`
- Report: `output/eqr_maze_probe_fs1_seed52.md`
- Visual casebook: `output/visualizations/index.html`
- Cases JSON: `output/visualizations/cases.json`
- Logs: `logs/launch.log`, `logs/run.log`
