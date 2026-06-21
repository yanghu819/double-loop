# RWKV Maze Boundary Objective: FutureSeed vs No FutureSeed

Source commit: `3df9e0275d378b7ad5b92c8da8f518df3a1db94a`

## Question

Can a generic PATH/non-PATH boundary objective turn the official Maze broad-mask shortcut into real pruning, and does FutureSeed help a causal RWKV backbone once broad coverage is penalized?

## Result

| Condition | loop8 path F1 | precision | recall | pred PATH frac | FP/case | FN/case | loop gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| no FutureSeed | 0.4089 | 0.3256 | 0.5828 | 0.2360 | 143.2 | 49.7 | +0.0004 |
| FutureSeed | 0.4278 | 0.3270 | 0.6280 | 0.2527 | 152.9 | 44.3 | +0.0008 |

FutureSeed delta: `+0.0189` path F1, mostly from better recall under pruning pressure. This is useful but weak: it does not beat the older broad-mask objective, and it does not prove loop correction.

## Interpretation

The objective answered one question: broad masks can be penalized without maze-specific repair or search. But the new failure mode is overpruning true path cells. FutureSeed is slightly more robust to that pressure, but loop1 and loop8 remain almost the same operating point.

Decision: discard this objective as a final benchmark, keep the result as mechanism evidence that FutureSeed can preserve useful signal under generic pruning pressure.

## Visualizations

- `comparison_summary.png`: aggregate training and final metrics.
- `hard_case_boundary_grid.png`: hard-case FP/FN overlays.
- `index.html`: local dashboard linking both condition-specific case viewers.
