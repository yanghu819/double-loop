# Maze31 Loop Self-Correction Probe

Run: `maze31-selfcorr-w1m002-boundary-d256l4-s800-20260618T0350Z-86604ee`

## Hypothesis

Earlier probes showed that candidate diversity and explicit budget/boundary state have capacity, but the training pressure still leaves the model at a broad-mask operating point. This probe changes only the training signal: loop1 defines the broad high-recall candidate mask, and later loops are directly trained to lower PATH probability on loop1 false-positive cells while preserving PATH probability on true-path cells.

This is more direct than CE, path-mass/floor pressure, or budget calibration because it supervises the relative change from early loop to later loop. The mechanism remains generic: no maze rules, search, repair, selector, topology prior, seed sweep, threshold sweep, or new head.

## Configuration

- Source SHA: `86604ee43be9cf3ef6a61f55e4f484f603b8094a`
- GPU: GPU1 only, `CUDA_VISIBLE_DEVICES=0`, A100 80GB
- Task: Maze31 perfect mazes, path length `160-260`
- Model: `state_compete_boundary`, hidden `256`, layers `4`, heads `8`
- Training: `800` steps, batch `16`, train loops `6`, eval loops `12`
- Objective: `EQR_SELF_CORRECTION_WEIGHT=1.0`, `EQR_SELF_CORRECTION_MARGIN=0.02`, `EQR_SELF_CORRECTION_START_LOOP=4`
- Path-mass pressure: disabled

## Results

| metric | loop1 | loop12 | delta |
|---|---:|---:|---:|
| path F1 | 0.5844 | 0.5851 | +0.0007 |
| precision | 0.4152 | 0.4165 | +0.0013 |
| recall | 0.9981 | 0.9952 | -0.0029 |
| predicted PATH frac | 0.4644 | 0.4617 | -0.0028 |
| label exact | 0.0000 | 0.0000 | +0.0000 |

Self-correction diagnostics at loop12:

| diagnostic | value |
|---|---:|
| self-correction loss | 0.0414 |
| false-positive candidate PATH prob delta | +0.0007 |
| true-path PATH prob delta | +0.0027 |
| hard FP count delta vs loop1 | -2.1113 |
| hard FN count delta vs loop1 | +0.5391 |
| boundary threshold mean | -0.4878 |
| boundary prune flip frac | 0.0000 |
| boundary add flip frac | 0.0432 |

Hard visualized cases:

| hard-case average | loop1 | loop12 |
|---|---:|---:|
| false positives | 286.5 | 285.0 |
| false negatives | 1.0 | 2.3 |

## Decision

This is another useful negative result, but it is sharper than the budget-state probe. The new signal is mechanically learnable: training self-correction loss moves from `0.0500` at step100 to `0.0412` at step800. It also moves the hard mask slightly in the intended direction: predicted PATH fraction drops and hard false positives fall.

But it does not create clean loop-time correction. The direct false-positive candidate probability does not decrease; at loop12 it is slightly higher than loop1 (`+0.0007`). The hard-mask improvement comes with more misses, not reliable pruning. The boundary module still learns a negative threshold and only adds PATH flips (`prune_flip=0.0`, `add_flip=0.0432`).

The main lesson is that a loop1-relative soft probability objective is still too easy to satisfy as calibration. FutureSeed plus loop is producing a broad useful hypothesis, but the later-loop training target must create a stronger discrete or distributional pressure for sparse correctness. Do not sweep self-correction weight, margin, seed, or start loop.

## Artifacts

- Metrics: `output/eqr_maze_probe_fs1_seed52.json`
- Report: `output/eqr_maze_probe_fs1_seed52.md`
- Visual casebook: `output/visualizations/index.html`
- Cases JSON: `output/visualizations/cases.json`
- Logs: `logs/launch.log`, `logs/run.log`
