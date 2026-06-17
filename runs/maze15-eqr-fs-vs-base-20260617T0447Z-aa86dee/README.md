# EqR Maze 15x15 FutureSeed vs Base Probe

- Queue: `maze15-eqr-fs-vs-base-20260617T0447Z-aa86dee`
- Source SHA: `aa86dee3dedcfe9cf7ee40e02628ad78d2816c09`
- GPU: GPU1 A100 80GB only
- Task: online perfect maze path-mask prediction
- Grid: `15x15`
- Path range: `32-56`
- Steps: `600`
- Batch: `128`
- Train loops: `4`
- Eval loops: `8`

## Question

Does the FutureSeed plus loop idea transfer from Sudoku to an EqR-style maze
proxy where the model must propagate path information over a grid?

## Results

| run | future_seed_scale | loop1 path_f1 | loop8 path_f1 | loop gain | exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `maze15-fs-s600-20260617T0447Z-aa86dee` | 1.0 | 0.590482 | 0.591242 | +0.000760 | 0.000000 |
| `maze15-base-s600-20260617T0449Z-aa86dee` | 0.0 | 0.590446 | 0.590488 | +0.000042 | 0.000000 |

## Interpretation

This is a successful task-plumbing run, not positive mechanism evidence.
The model learned a rough path mask, but both variants over-predict PATH:

- FutureSeed loop8 path label fraction: `0.178464`
- FutureSeed loop8 path prediction fraction: `0.420851`
- FutureSeed loop8 precision/recall: `0.423622` / `0.999009`

So the first maze setting mostly tests a broad path-highlighter behavior. It
does not yet force the model to use recurrent correction, and the FutureSeed
delta is within noise.

## Decision

Do not claim FutureSeed transfers to maze from this run. The next maze run
should increase recurrence pressure or improve the target calibration so that
loop updates have a chance to matter. A useful follow-up is a harder/longer maze
probe with the same two-arm comparison, or a diagnostic run that tracks whether
later loops reduce over-prediction rather than merely keeping recall high.

