# Maze31 Readout-Gated State Dynamics Probe

Run: `maze31-readoutgate-d256l4-s800-20260617T132624Z-e4e8afd`

Source SHA: `e4e8afd1f31076af022df4cfb4fe2ecce3644e3f`

## Question

Fixed delta state updates only changed coverage. Can a minimal learned gate,
trained before the readout, learn per-token keep/prune/revise behavior and make
later loops continue correcting Maze31 path masks?

## Configuration

- Task: 31x31 perfect mazes, hard path range 160-260
- Model: hidden 256, 8 heads, 4 layers
- Train/eval loops: 6/12
- Steps: 800
- State update: `learned_gate`, gate bias `2.0`
- FutureSeed scale: 1
- Noise: none

## Result

| loop | exact | path F1 | precision | recall | pred path frac | H gate mean | H gate std | L gate mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.5850 | 0.4155 | 0.9997 | 0.4648 | 0.8960 | 0.0490 | 0.8789 |
| 12 | 0.0000 | 0.5851 | 0.4157 | 0.9996 | 0.4646 | 0.8758 | 0.0122 | 0.8789 |

The gate is now trainable: H gate mean and std move away from the initialization.
But the mechanism still does not create useful later-loop correction. Loop gain is
only `+0.0001`, and the model over-predicts PATH cells even more than the fixed
delta run.

## Decision

This is a useful negative mechanism result. A minimal per-token gate can learn,
but it learns a high-recall broad mask, not pruning or exact path consistency.
Do not sweep gate bias. The next state-dynamics idea needs a richer learned
update that can compare local alternatives or maintain path-continuity evidence,
while staying generic and not using hand-written maze repair.
