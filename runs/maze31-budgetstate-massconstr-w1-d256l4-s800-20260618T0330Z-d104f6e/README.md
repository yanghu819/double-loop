# Maze31 Budget-State Boundary Probe

Run: `maze31-budgetstate-massconstr-w1-d256l4-s800-20260618T0330Z-d104f6e`

## Hypothesis

The previous simultaneous constrained path-mass objective was mechanically active but too weak to move the hard PATH mask. This probe adds a minimal recurrent decision variable: a learned per-sample, per-loop budget shift on top of the generic boundary calibrator. The shift sees only generic features: PATH margin, soft PATH mass, entropy, candidate disagreement, candidate weights, and loop index.

Prediction: if the bottleneck was missing boundary capacity, loop12 should reduce predicted PATH fraction and false positives while keeping recall close to loop1. If it still expands or only makes tiny calibration trades, the bottleneck is the training pressure/objective, not the absence of a boundary state.

No maze rules, search, repair, selector, seed sweep, threshold sweep, or hand-coded topology prior were used.

## Configuration

- Source SHA: `d104f6ec43e0fbd510705923231b4655b0d50937`
- GPU: GPU1 only, `CUDA_VISIBLE_DEVICES=0`, A100 80GB
- Task: Maze31 perfect mazes, path length `160-260`
- Model: `state_compete_budget`, hidden `256`, layers `4`, heads `8`
- Training: `800` steps, batch `16`, train loops `6`, eval loops `12`
- Objective: constrained path-mass, `EQR_PATH_MASS_WEIGHT=1.0`, `EQR_PATH_MASS_START_LOOP=4`

## Results

| metric | loop1 | loop12 | delta |
|---|---:|---:|---:|
| path F1 | 0.5852 | 0.5861 | +0.0008 |
| precision | 0.4168 | 0.4184 | +0.0016 |
| recall | 0.9938 | 0.9897 | -0.0041 |
| predicted PATH frac | 0.4606 | 0.4570 | -0.0036 |
| non-PATH PATH prob | 0.2439 | 0.2447 | +0.0008 |
| true-PATH PATH prob | 0.7317 | 0.7349 | +0.0031 |
| positive floor violation frac | 0.0000 | 0.3937 | +0.3937 |

Boundary diagnostics at loop12:

| diagnostic | value |
|---|---:|
| budget shift mean | -0.0753 |
| threshold mean | -0.4865 |
| raw hard PATH frac | 0.4380 |
| calibrated hard PATH frac | 0.4580 |
| raw soft PATH frac | 0.2969 |
| calibrated soft PATH frac | 0.3398 |
| prune flip frac | 0.0000 |
| add flip frac | 0.0200 |
| keep/proposed/context weights | 0.3757 / 0.4438 / 0.1806 |
| context minus proposed RMS | 1.2037 |

Hard visualized cases:

| hard-case average | loop1 | loop12 |
|---|---:|---:|
| false positives | 283.4 | 282.2 |
| false negatives | 3.3 | 5.6 |

## Decision

This is a useful negative result. The budget state was trainable and did not collapse the candidate mixer, but it learned the wrong operating direction. The learned threshold is negative, so calibration adds PATH mass instead of pruning it: `prune_flip=0.0`, `add_flip=0.0200`. The apparent F1 gain is only `+0.0008`, and hard cases show the same failure shape as before: false positives barely drop while false negatives rise.

Compared with the previous constrained path-mass run, budget state increases boundary expressivity but does not solve recall-preserving pruning. The decisive diagnostic is that a learned budget variable still follows the same broad-mask training pressure. The bottleneck is now more likely objective alignment: loops need a training signal that rewards a later state for moving from broad coverage to correct sparse path decisions without relying on hand-coded maze structure.

Do not sweep budget scale, seed, threshold bias, or margin. The next high-ROI direction should either simplify back to the strongest positive signal, late-loop mass pressure, and make its recall preservation more direct, or switch to a generic denoising/self-correction setup where later loops must recover held-out true positives while suppressing distractor mass.

## Artifacts

- Metrics: `output/eqr_maze_probe_fs1_seed52.json`
- Report: `output/eqr_maze_probe_fs1_seed52.md`
- Visual casebook: `output/visualizations/index.html`
- Cases JSON: `output/visualizations/cases.json`
- Logs: `logs/launch.log`, `logs/run.log`
