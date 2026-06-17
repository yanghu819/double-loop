# Maze31 Density-Bridge Curriculum Abort

Run: `maze31-densitybridge-d192-s2400-20260617T084558Z-d5a8258`

Source SHA: `d5a82583fd3e4f160cf86731c05f35a1bc3fc445`

## Question

Was the failed 80-140 warmup only caused by path labels being too sparse? Try a
tighter bridge, `120-200 -> 160-260`, without changing the model or loss.

## Configuration

- Model: hidden 192, 6 heads, 2 layers
- Train loops/eval loops: 6/12
- Planned stages: `120-200:600,160-260:1800`
- Batch: 32
- Path loss weight: 1.5

## Result

Stopped at step400 because the model collapsed even more strongly toward
non-PATH predictions:

| step | path range | CE | train path F1 | exact |
|---:|---|---:|---:|---:|
| 100 | 120-200 | 0.3496 | 0.0605 | 0.0000 |
| 200 | 120-200 | 0.3457 | 0.0377 | 0.0000 |
| 300 | 120-200 | 0.3438 | 0.0150 | 0.0000 |
| 400 | 120-200 | 0.3438 | 0.0164 | 0.0000 |

## Decision

Abort. Maze31 path-length curriculum is not the current high-ROI axis. The
failed runs teach that hard-from-start at least learns a broad path mask, while
shorter warmups do not even preserve that behavior.
