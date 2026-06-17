# Maze31 Context Improvement Probe

Run: `maze31-cross-improve-w01m001-d256l4-s800-20260617T184714Z-ebb198d`

## Hypothesis

The previous temporal predictive target failed because it asked the context
candidate to copy the next broad mask. This run instead asked context logits to
beat the detached current logits on currently wrong tokens. If the state
dynamics bottleneck is a missing correction direction, this target should create
loop-time pruning: lower predicted path fraction, higher precision, and positive
loop gain from loop1 to loop12.

## Configuration

- GPU: AIStation GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Git SHA: `ebb198d6d17dca9fd448a1d47ff95ab220451dee`
- Task: Maze31 perfect mazes, path length 160-260
- Model: EqR + FutureSeed, hidden 256, layers 4, heads 8
- Loop setup: train loops 6, eval loops 12, all-loop supervised loss
- State update: `state_compete_cross`
- New objective: `context_improve_weight=0.1`, `context_improve_margin=0.01`
- No maze rules, no repair/search/selector, no seed/bias/temp sweep

## Result

| metric | loop1 | loop12 |
|---|---:|---:|
| path F1 | 0.5836 | 0.5830 |
| precision | 0.4144 | 0.4135 |
| recall | 0.9982 | 1.0000 |
| predicted path fraction | 0.4654 | 0.4672 |
| exact | 0.0000 | 0.0000 |

Loop gain: `-0.0006`.

The objective was mechanically satisfied but did not change behavior. At loop12,
`context_improve_loss=0.0`, and context CE on current errors was lower than
current CE by about `0.590`. Candidate competition also stayed non-collapsed:
keep/proposed/context means were `0.366/0.323/0.311`, and
`context_minus_proposed_rms=0.950`.

## Failure Shape

The visual failures remain broad-mask failures. In the selected hard cases,
loop1 already covers nearly all true path cells, but carries hundreds of false
positive path cells. Loop12 preserves or slightly expands those false positives.
For the first case, false positives move from `288` at loop1 to `290` at
loop12, with zero misses in both loops.

## Decision

Discard this objective as the next mainline. It improves confidence on the
current error tokens but does not teach the model which alternative open cells
should be pruned. The next high-ROI direction should keep the generic
FutureSeed + loop framing, but use a sharper correction signal tied to mask
sharpening, uncertainty, or contrast between retained and rejected cells rather
than token CE improvement alone.
