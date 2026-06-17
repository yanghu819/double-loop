# Maze31 Joint Width+Depth Scale Probe

Run: `maze31-jointscale-d256l4-s1200-20260617T093319Z-a64efac`

Source SHA: `a64efac5c206dd2a33e3cb42c7e94dd8af602498`

## Question

Width alone did not break the early plateau, depth alone did not make loops
revise state, and path-length curriculum was actively harmful. Does a simple
joint width+depth scale-up cross the Maze31 threshold?

## Configuration

- GPU row: GPU1 only
- Task: 31x31 perfect mazes
- Path range: 160-260 from step 1
- Model: hidden 256, 8 heads, 4 layers
- Parameters: 3,542,532
- Train loops/eval loops: 6/12
- Train steps: 1200
- Batch/eval N: 32/512
- FutureSeed scale: 1
- Path loss weight: 1.5

## Result

| loop | exact | path F1 | precision | recall | predicted path fraction |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.5662 | 0.4384 | 0.8085 | 0.3563 |
| 12 | 0.0000 | 0.5649 | 0.4376 | 0.8061 | 0.3559 |

Loop gain is `-0.0013`. The model finds more true path cells than D192/L4, but
does so by predicting far too many PATH cells. It still does not exact-solve,
and later loops do not clean the mask.

## Decision

This is a small local mask-quality improvement, not a recurrence breakthrough.
Do not continue clean Maze31 scale by filling D/L tables. The next high-ROI
direction is a simple loop/FutureSeed state-dynamics change that lets later
loops revise over-broad path hypotheses.
