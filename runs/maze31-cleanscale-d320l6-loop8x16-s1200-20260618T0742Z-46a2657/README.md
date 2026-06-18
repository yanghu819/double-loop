# Maze31 Clean D320/L6 Full Readout

Recorded: 2026-06-18T08:31Z

## Hypothesis

The previous matched pair gave a useful fork: `state_compete_cross` at D320/L6
failed to open PATH prediction through step400, while the clean FutureSeed +
loop fallback opened at step400 but was killed before eval. This run asks the
cleaner scaling question: if we spend the same generic capacity on a simple
FutureSeed backbone plus recurrent loops, do later loops become useful
correction, or do they only add weak coverage?

This is intentionally not a repair, selector, search, scalar-loss sweep, or
maze-rule intervention.

## Config

- Remote row: GPU1 only, A800 80GB, `CUDA_VISIBLE_DEVICES=0`
- Git SHA: `46a2657df6ad18b3994f5f81920bf3c3d966e68b`
- Task: Maze31 perfect mazes, path length `160-260`
- Model: FutureSeed + loop, hidden `320`, layers `6`, heads `8`
- State update: `none`
- Train/eval loops: `8 / 16`
- Steps/batch/eval: `1200 / 16 / 512`
- Loss: standard CE plus `MAZE_PATH_LOSS_WEIGHT=1.5`
- No feature noise, no state competition, no maze repair/search/selector

## Result

The model is trainable, but recurrence does not become a strong correction
process.

| metric | loop1 | loop16 | delta |
|---|---:|---:|---:|
| path F1 | 0.5301 | 0.5344 | +0.0044 |
| precision | 0.4431 | 0.4434 | +0.0003 |
| recall | 0.6664 | 0.6794 | +0.0130 |
| predicted PATH fraction | 0.2907 | 0.2961 | +0.0054 |
| exact | 0.0000 | 0.0000 | +0.0000 |

Train curve:

| step | CE | train path F1 |
|---:|---:|---:|
| 100 | 1.0547 | 0.0000 |
| 200 | 1.0703 | 0.0000 |
| 300 | 1.0625 | 0.0000 |
| 400 | 0.3691 | 0.5862 |
| 800 | 0.3652 | 0.4824 |
| 1200 | 0.3652 | 0.4226 |

Visual hard-case summary over 16 generated cases:

| metric | loop1 | loop16 | delta |
|---|---:|---:|---:|
| path F1 | 0.3416 | 0.3455 | +0.0039 |
| precision | 0.2755 | 0.2773 | +0.0018 |
| recall | 0.4675 | 0.4767 | +0.0092 |
| predicted PATH fraction | 0.2915 | 0.2953 | +0.0038 |
| false positives | 202.9 | 205.1 | +2.1 |
| false negatives | 88.1 | 86.6 | -1.5 |

The loop effect is therefore mostly coverage: it fills a few missing true-path
cells, but it also adds false positives. It does not behave like pruning.

## Decision

Do not continue this exact clean D320/L6 Maze31 setup just by adding more
loop-depth or repeating the same 1200-step budget. The result is useful because
it separates two facts:

1. Clean scaling is optimizable; the earlier cross-candidate failure was not a
   general D320/L6 impossibility.
2. Clean scaling alone did not create the late-loop self-correction needed for
   hard Maze31.

The next high-ROI direction should keep the bitter-lesson constraint but change
the recurrent decision/state dynamics in a simple way. The target is not a new
maze-specific rule; it is a generic mechanism that lets later loops move from
"add more plausible PATH" to "revise the current decision boundary while
preserving true-path evidence."

## Artifacts

- `config.json`, `metadata.json`, `score.json`
- `logs/run.log`, `logs/launch.outer.log`
- `output/eqr_maze_probe_fs1_seed52.json`
- `output/eqr_maze_probe_fs1_seed52.md`
- `output/visualizations/index.html`
- `output/visualizations/casebook.md`
- `output/visualizations/cases.json`
- `output/visualizations/inline/summary_loop_effect.png`
- `output/visualizations/inline/case_best_loop_gain.png`
- `output/visualizations/inline/case_worst_loop_regression.png`
- `output/visualizations/inline/case_typical_failure.png`
- `visualizations/index.html`
- `visualizations/summary.json`
- `source_HEAD.txt`, `source.patch`, `source_snapshot.ls.txt`
