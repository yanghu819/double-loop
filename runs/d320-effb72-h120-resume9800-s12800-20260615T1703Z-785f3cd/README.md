# D320 Effective-Batch h120 Interrupted Segment

This segment resumed the D320 effective-batch h120 run from the step9800/10000 checkpoint and was interrupted by a GPU1 halt before the first planned eval at step10800.

It is not a model-quality result.

## Observed Progress

- Source SHA: `785f3cdca74aab1f6e9d6a4cc252a17e3aa63731`
- Last logged step: 10600
- Last saved train checkpoint: `checkpoints/latest.pt` at step10000
- Logged CE after resume:
  - step9900: 0.2925
  - step10000: 0.3120
  - step10100: 0.2566
  - step10200: 0.2802
  - step10300: 0.2958
  - step10400: 0.3764
  - step10500: 0.3454
  - step10600: 0.3155

## Decision

Treat as platform interruption, not model failure. The experiment was resumed from the periodic checkpoint as `d320-effb72-h120-resume10000-s12800-20260615T1801Z-785f3cd`.
