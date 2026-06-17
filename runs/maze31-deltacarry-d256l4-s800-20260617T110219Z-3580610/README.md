# Maze31 Delta-Carry State Dynamics Probe

Run: `maze31-deltacarry-d256l4-s800-20260617T110219Z-3580610`

Source SHA: `3580610a3d716650567ac42f87ce11e114c19d84`

## Question

Maze31 D256/L4 can cover much of the true path but predicts far too many PATH
cells, and later loops do not prune them. Does explicitly carrying a fraction
of the previous state correction give loop a direction that enables continued
revision?

## Configuration

- Task: 31x31 perfect mazes, hard path range 160-260
- Model: hidden 256, 8 heads, 4 layers
- Train/eval loops: 6/12
- Steps: 800
- State update: `delta_carry`
- Delta scale/decay: `0.35` / `0.95`
- FutureSeed scale: 1

## Result

| loop | exact | path F1 | precision | recall | pred path frac |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.5709 | 0.4376 | 0.8299 | 0.3663 |
| 12 | 0.0000 | 0.5721 | 0.4372 | 0.8367 | 0.3697 |

Loop gain becomes slightly positive (`+0.0012`) instead of negative, and F1 is
slightly higher than the clean D256/L4 score (`0.5649`). But the gain comes
from more recall and more predicted PATH mass, not from pruning false branches.

## Decision

Positive delta-carry is a weak positive signal for state dynamics but not a
solution. It amplifies the over-coverage failure mode. The next state update
should be learned/gated, not another fixed positive momentum run.
