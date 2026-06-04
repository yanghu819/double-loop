# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0057
- train_sec: 288.9

## Clean Eval

- holes16: exact=0.9980, blank=0.9998, valid=0.9980
- holes20: exact=0.9844, blank=0.9977, valid=0.9883
- holes24: exact=0.9219, blank=0.9915, valid=0.9316
- holes28: exact=0.7090, blank=0.9690, valid=0.7246
- holes32: exact=0.3145, blank=0.9097, valid=0.3184

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
