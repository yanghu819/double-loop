# Maze31 Dense Context Ranking Probe

Run: `maze31-cross-denserank-w005m025-d256l4-s800-20260617T204516Z-cc01d1d`

## Hypothesis

Hard min/max ranking was aligned with pruning but too sparse: it only trained
the weakest true-path cell against the strongest false-positive PATH cell. This
run uses dense pairwise ranking, so every true-path versus false-positive pair
inside the current predicted PATH candidate set contributes gradient.

If sparse supervision was the bottleneck, dense ranking should create a stronger
loop-time pruning signal: loop12 should be narrower than loop1, precision should
rise, and recall should not collapse.

## Configuration

- GPU: AIStation GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Git SHA: `cc01d1d4cf2b83446801d8ca79137208fa6b5e51`
- Task: Maze31 perfect mazes, path length 160-260
- Model: EqR + FutureSeed, hidden 256, layers 4, heads 8
- Loop setup: train loops 6, eval loops 12, all-loop supervised loss
- State update: `state_compete_cross`
- Dense ranking: `context_rank_weight=0.05`, `context_rank_margin=0.25`, `context_rank_mode=dense`
- Disabled auxiliary copy/improvement objectives: predictive weight `0`, CE-improve weight `0`
- No maze rules, no repair/search/selector, no seed/bias/temp sweep

## Result

| metric | loop1 | loop12 |
|---|---:|---:|
| path F1 | 0.5854 | 0.5857 |
| precision | 0.4162 | 0.4166 |
| recall | 0.9984 | 0.9982 |
| predicted path fraction | 0.4634 | 0.4629 |
| exact | 0.0000 | 0.0000 |

Loop gain: `+0.0003`.

Dense ranking did improve the diagnostic score separation. At loop12:

- positive PATH score: `4.0795`
- false-positive PATH score: `4.0246`
- mean margin: `+0.0548`
- hard margin: `-1.2340`
- dense ranking loss: `0.8152`

Candidate competition stayed non-collapsed:
keep/proposed/context means were `0.400/0.460/0.141`, and
`context_minus_proposed_rms=1.060`.

## Failure Shape

The broad-mask failure remains. In the 10 visualized hard cases, false positives
move only from `286.4` to `286.2` on average. The worst false-positive cells
still survive, which matches the very negative hard margin. Dense ranking makes
the average true-path score higher than the average false-positive score, but
the output threshold and recurrent state do not turn that average preference
into decisive pruning.

## Decision

This is a useful mechanistic result but not a success. Dense ranking is more
optimizable than hard ranking and creates a real average preference, but average
preference is insufficient for exact path pruning. Do not sweep rank
weight/margin/seed. The next high-ROI direction is to expose uncertainty or
candidate contrast inside the recurrent state update itself, or to use a simple
learned operating-threshold/state-normalization mechanism so loops can convert
score gaps into actual mask changes.
