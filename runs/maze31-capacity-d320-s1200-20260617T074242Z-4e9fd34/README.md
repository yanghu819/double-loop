# Maze31 D320 Capacity Probe Abort

Run: `maze31-capacity-d320-s1200-20260617T074242Z-4e9fd34`

Source SHA: `4e9fd341eca88c04e3e2f6dca3d68f476b72acdc`

Date: 2026-06-17 UTC

## Question

Is the Maze31 failure from the hidden-192 frontier run mainly a representation
capacity issue? If yes, widening to hidden 320 should quickly move the training
curve above the D192 plateau and make a full final eval worth waiting for.

This is not a table-filling width run. It is a kill-or-continue probe for one
decision: keep spending on raw capacity, or change the data/curriculum/state
dynamics axis.

## Configuration

- GPU row: GPU1 only
- Task: 31x31 perfect mazes
- Path length range: 160-260 from step 1
- Model: hidden 320, 8 heads, 2 layers
- Train loops: 6
- Planned eval loops: 12
- Planned train steps: 1200
- Batch: 32
- FutureSeed scale: 1
- Path loss weight: 1.5

## Abort

The run was interrupted by the GPU1 lease halt after step600, before final eval
or visualization export. It is therefore not a scored model-quality result.

See `abort.json` for the structured abort record and `logs/run.log` for the
training curve.

## Observed Curve

| step | CE | train path F1 | exact |
|---:|---:|---:|---:|
| 100 | 0.3574 | 0.5570 | 0.0000 |
| 200 | 0.3516 | 0.5814 | 0.0000 |
| 300 | 0.3574 | 0.5588 | 0.0000 |
| 400 | 0.3496 | 0.5158 | 0.0000 |
| 500 | 0.3496 | 0.5551 | 0.0000 |
| 600 | 0.3555 | 0.5560 | 0.0000 |

The early curve is essentially the same as the D192 Maze31 plateau rather than
a clear capacity breakout.

## Decision

Do not rerun this exact D320 setup just to obtain a final eval. The useful
signal already arrived: wider hidden size by itself did not quickly change the
optimization behavior when trained only on hard 160-260 paths.

The next high-ROI test is a Maze31 curriculum bridge: start from shorter
31x31 paths, then move to 160-260. That keeps the bitter-lesson direction
intact because it changes the data scale/optimization schedule, not by adding a
handwritten path repair rule or maze-specific solver.
