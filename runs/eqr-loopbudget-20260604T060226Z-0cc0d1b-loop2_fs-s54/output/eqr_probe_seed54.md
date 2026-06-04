# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 1.0
- train_ce: 0.0210
- train_sec: 292.2

## Clean Eval

- holes16: exact=0.9980, blank=0.9994, valid=0.9980
- holes20: exact=0.9902, blank=0.9985, valid=0.9902
- holes24: exact=0.9316, blank=0.9910, valid=0.9395
- holes28: exact=0.7520, blank=0.9667, valid=0.7656
- holes32: exact=0.4102, blank=0.9080, valid=0.4199

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
