# Maze21 Recurrence-Pressure Probe

- Queue: `maze21-pressure-fs-vs-base-20260617T0525Z-ca0f837`
- Source SHA: `ca0f83709963cbf25b60f75eec9f2abd0d173404`
- GPU: GPU1 A100 80GB only
- Task: online perfect maze path-mask prediction
- Grid: `21x21`
- Path range: `80-140`
- Path loss weight: `1.5`
- Hidden/layers/heads: `192/2/6`
- Train/eval loops: `6/10`
- Steps per arm: `1200`
- Batch: `64`

## Question

The 15x15 probe showed broad path over-prediction and almost no loop effect.
This run asks whether a harder maze with better target calibration creates real
recurrent correction, and whether FutureSeed helps beyond a base recurrent EqR
state.

## Results

| run | future_seed_scale | loop1 path_f1 | loop10 path_f1 | loop gain | loop10 pred_frac | loop10 precision | loop10 recall | exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `maze21-pressure-fs-s1200-20260617T0525Z-ca0f837` | 1.0 | 0.687345 | 0.805056 | +0.117710 | 0.279682 | 0.703097 | 0.950226 | 0.000000 |
| `maze21-pressure-base-s1200-20260617T0525Z-ca0f837` | 0.0 | 0.709555 | 0.761380 | +0.051825 | 0.319808 | 0.630754 | 0.971706 | 0.000000 |

The label path fraction is `0.207045` for both evals.

## Interpretation

This is the first maze run where loop compute clearly matters. FutureSeed does
not merely increase recall; by loop10 it reduces over-prediction more strongly
than the base model and gets a higher F1:

- FutureSeed path F1 advantage: `+0.043675`
- FutureSeed loop gain advantage: `+0.065885`
- FutureSeed loop10 predicted path fraction is closer to the label fraction
  (`0.2797` vs base `0.3198`, label `0.2070`)
- FutureSeed loop10 precision is higher (`0.7031` vs `0.6308`) while recall
  remains high (`0.9502`)

This supports the mechanism claim that FutureSeed plus loop can improve
iterative global path refinement on a non-Sudoku proxy. It does not yet prove
full discrete path solving, because exact remains `0.0` on held-out eval.

## Decision

Continue the maze proxy, but do not do seed-table filling. The next useful
experiment should either push Maze21 to exact/path_exact with a longer run or
increase path/grid pressure while keeping this calibrated target. If exact stays
zero with high F1, the next modeling question is how to make loop updates
sharpen a single path rather than only reduce a broad mask.

