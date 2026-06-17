# D320 Learned-Gate Continuation Early Stop

- Run: `d320-updategate-cont-s26600-20260617T0400Z-424d395`
- Source SHA: `424d395851a46481cd0252aa84ac0b7981a98f9a`
- GPU: GPU1 A100 80GB only
- Resume checkpoint: learned-gate D320 step23600 run
- Stop point: checkpoint eval at step24600

## Result

The h120 primary readout at step24600 was:

- `eval_by_holes.holes120.eval_clean.loop6.label_exact = 0.328125`
- `blank_acc = 0.8612467647`
- train CE at step24600: `0.061891`
- learned update gates: `H=0.99108`, `L=0.99235`

This is below the prior learned-gate checkpoint score (`0.349609`) and does not
beat the prior final full-eval score (`0.333984`). The run was stopped to avoid
spending GPU on a same-direction continuation with low marginal information.

## Decision

Do not continue this exact learned-gate checkpoint just by adding more steps.
The useful next work is either a genuinely different simple state update or a
different effective scaling axis. Selector, Sudoku repair, delayed loop-credit,
feature-noise tables, and Sudoku-specific priors remain low ROI.

