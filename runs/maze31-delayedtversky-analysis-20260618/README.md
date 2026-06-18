# Maze31 Delayed Tversky Analysis

Recorded: 2026-06-18T11:10Z

## Question

Can delayed generic FP/FN pressure make FutureSeed+loop satisfy the second
criterion: later loops reduce false positives without adding false negatives?

## Runs

- Clean FutureSeed:
  `maze31-cleanscale-d320l6-loop8x16-s1200-20260618T0742Z-46a2657`
- Always-on Tversky:
  `maze31-fs-tversky-w075-a075b065-d320l6-loop8x16-s1200-20260618T091423Z-d19cd3a`
- Delayed Tversky:
  `maze31-fs-delayedtversky-a450-w075-a075b065-d320l6-loop8x16-s1200-20260618T103230Z-5009fc1`

## Result

Delayed Tversky preserves opening and improves final score:

- Clean final loop16 path F1: `0.5344`
- Delayed final loop16 path F1: `0.5824`
- Always-on Tversky: stayed at `0.0` through step600 and was stopped

But delayed Tversky does not create loop-time repair:

- Delayed loop1 to loop16 F1: `0.5822 -> 0.5824`
- Precision: `0.4305 -> 0.4308`
- Recall: `0.9096 -> 0.9091`
- Predicted PATH fraction: `0.4082 -> 0.4077`
- Visual FP/FN: `260.5/30.8 -> 260.2/31.3`

## Interpretation

This is a useful positive/negative split.

Positive: correction pressure must be delayed until after FutureSeed opening.
This is a real training lesson and raises the final operating point.

Negative: the loop itself is still almost a copy operation. The final score
improvement comes from training the model into a different mask regime, not from
later loops repeatedly pruning false positives.

## Decision

Do not run Tversky timing/weight sweeps. Keep delayed pressure as a possible
training guard, but move the main mechanism work to a generic recurrent
state/decision update that gives later loops a reason and a signal to revise the
mask.

## Artifacts

- `training_curve.png`
- `loop_metric_comparison.png`
- `comparison.json`
