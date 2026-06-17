# Maze31 Anti-Delta State Dynamics Probe

Run: `maze31-antidelta-d256l4-s600-20260617T113214Z-3580610`

Source SHA: `3580610a3d716650567ac42f87ce11e114c19d84`

## Question

Positive delta-carry slightly improves F1 but increases over-predicted PATH
mass. If the problem is over-coverage, does a negative delta damp the carry and
improve precision/pruning?

## Configuration

- Task: 31x31 perfect mazes, hard path range 160-260
- Model: hidden 256, 8 heads, 4 layers
- Train/eval loops: 6/12
- Steps: 600
- State update: `delta_carry`
- Delta scale/decay: `-0.35` / `0.95`
- FutureSeed scale: 1

## Result

| loop | exact | path F1 | precision | recall | pred path frac |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.3373 | 0.4550 | 0.2697 | 0.1145 |
| 12 | 0.0000 | 0.3481 | 0.4526 | 0.2846 | 0.1217 |

Anti-delta reduces predicted PATH mass, but it collapses recall. The resulting
path F1 is far below both clean D256/L4 and positive delta-carry.

## Decision

Fixed-sign linear state extrapolation is not enough. Positive sign over-covers;
negative sign under-covers. The useful next mechanism is a small learned/gated
state update that can decide when to keep, prune, or revise, rather than a
constant carry transformation.
