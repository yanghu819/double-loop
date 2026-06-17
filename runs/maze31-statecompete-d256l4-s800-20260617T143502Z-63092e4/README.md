# Maze31 State-Competition Dynamics Probe

Run: `maze31-statecompete-d256l4-s800-20260617T143502Z-63092e4`

Source SHA: `63092e42c5b17d11d786f724496b387448ba2723`

## Question

Readout-gated state dynamics was trainable, but it still behaved like a
coverage knob. Can a minimal candidate-competition update let the recurrent
state compare local/global alternatives and create a pruning signal?

## Configuration

- Task: 31x31 perfect mazes, hard path range 160-260
- Model: hidden 256, 8 heads, 4 layers
- Train/eval loops: 6/12
- Steps: 800
- State update: `state_compete`
- Candidates: keep previous H state, accept proposed H state, accept
  context-competed H state
- FutureSeed scale: 1
- Noise: none

## Result

| loop | exact | path F1 | precision | recall | pred path frac | keep | proposed | context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.5875 | 0.4199 | 0.9893 | 0.4552 | 0.0137 | 0.9665 | 0.0195 |
| 12 | 0.0000 | 0.5874 | 0.4198 | 0.9898 | 0.4556 | 0.0191 | 0.9612 | 0.0194 |

Compared with the readout-gate run, this is a small move in the right direction:
predicted PATH fraction drops from `0.4646` to `0.4556`, and precision rises
from `0.4157` to `0.4198`. But the effect is present already at loop1. Loop12
does not improve over loop1 (`-0.0001` F1 gain), and the model still predicts
more than twice the true path mass (`0.4556` versus `0.1932`).

## Decision

This is not a loop-pruning success. Candidate competition is more promising
than scalar gating because it nudges the operating point toward precision, but
the current softmax stays dominated by the proposed state and does not create
later-loop correction. The next mechanism should make the competing candidate
stronger or train it under explicit recurrence pressure, not sweep seeds or
gate bias.
