# Maze31 Short-Path Curriculum Abort

Run: `maze31-currbridge-d192-s2400-20260617T083706Z-d5a8258`

Source SHA: `d5a82583fd3e4f160cf86731c05f35a1bc3fc445`

## Question

Can Maze31 open if the model first learns shorter 31x31 paths, then moves to
the hard 160-260 range?

This was a data scaling/optimization test, not a solver prior. Hard evaluation
would still have used 160-260.

## Configuration

- Model: hidden 192, 6 heads, 2 layers
- Train loops/eval loops: 6/12
- Planned stages: `80-140:600,120-200:600,160-260:1200`
- Batch: 32
- Path loss weight: 1.5

## Result

Stopped at step600. The first stage did not make the model learn a usable path
mask:

| step | path range | CE | train path F1 | exact |
|---:|---|---:|---:|---:|
| 100 | 80-140 | 0.3125 | 0.0000 | 0.0000 |
| 200 | 80-140 | 0.3047 | 0.0014 | 0.0000 |
| 300 | 80-140 | 0.2852 | 0.0884 | 0.0000 |
| 400 | 80-140 | 0.2969 | 0.0611 | 0.0000 |
| 500 | 80-140 | 0.2988 | 0.0810 | 0.0000 |
| 600 | 80-140 | 0.2910 | 0.1634 | 0.0000 |

## Decision

Abort. Short-path warmup makes PATH tokens too sparse and encourages
conservative non-PATH predictions. This rejects the naive easy-to-hard path
curriculum for Maze31.
