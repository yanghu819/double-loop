# Maze31 Context Path Ranking Probe

Run: `maze31-cross-rank-w005m025-d256l4-s800-20260617T194554Z-6a29936`

## Hypothesis

The CE-improvement target failed because it made context more confident without
teaching which predicted PATH cells should be pruned. This run uses a direct
ranking signal inside the current predicted PATH candidate set: true-path cells
should have higher context PATH scores than false-positive PATH cells.

This is closer to pruning than CE because it compares retained versus rejected
cells within the broad mask, instead of asking for better token confidence over
the whole board.

## Configuration

- GPU: AIStation GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Git SHA: `6a299368242fef6b8d53ea451a25df05c05db524`
- Task: Maze31 perfect mazes, path length 160-260
- Model: EqR + FutureSeed, hidden 256, layers 4, heads 8
- Loop setup: train loops 6, eval loops 12, all-loop supervised loss
- State update: `state_compete_cross`
- New objective: `context_rank_weight=0.05`, `context_rank_margin=0.25`
- Disabled auxiliary copy/improvement objectives: predictive weight `0`, CE-improve weight `0`
- No maze rules, no repair/search/selector, no seed/bias/temp sweep

## Result

| metric | loop1 | loop12 |
|---|---:|---:|
| path F1 | 0.5851 | 0.5852 |
| precision | 0.4167 | 0.4171 |
| recall | 0.9938 | 0.9923 |
| predicted path fraction | 0.4608 | 0.4596 |
| exact | 0.0000 | 0.0000 |

Loop gain: `+0.0001`.

Candidate competition remained non-collapsed at loop12:
keep/proposed/context means were `0.380/0.380/0.240`, and
`context_minus_proposed_rms=1.004`.

The ranking objective did not really open. Train rank loss stayed around
`0.83`, and eval hard margin stayed negative at loop12 (`-0.011`). The mean
positive and negative context PATH scores were almost identical
(`4.468784` versus `4.468800`), so the model did not learn a robust separation
between true path cells and false positives.

## Failure Shape

The output is still a broad-mask failure. The selected cases show small pruning
motion, but not enough to matter. Across the 10 visualized hard cases, false
positives average `285.1 -> 285.0`; recall drops slightly in some cases, and
exact remains zero.

## Decision

This is a weak directional signal, not a success. Ranking is better aligned
with pruning than CE advantage, because precision and predicted path fraction
move in the right direction. But the hard min/max ranking target is too sparse
or too hard for the current setup, so it does not create sustained loop-time
correction. Do not sweep seeds. The next high-ROI change is a smoother or denser
pruning signal, or a direct state-update mechanism that exposes uncertainty and
candidate contrast to later loops.
