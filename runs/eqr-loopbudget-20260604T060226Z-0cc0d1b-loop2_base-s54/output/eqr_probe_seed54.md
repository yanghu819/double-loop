# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0236
- train_sec: 289.6

## Clean Eval

- holes16: exact=1.0000, blank=1.0000, valid=1.0000
- holes20: exact=0.9883, blank=0.9992, valid=0.9883
- holes24: exact=0.9062, blank=0.9883, valid=0.9121
- holes28: exact=0.7168, blank=0.9667, valid=0.7305
- holes32: exact=0.3301, blank=0.8987, valid=0.3379

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
