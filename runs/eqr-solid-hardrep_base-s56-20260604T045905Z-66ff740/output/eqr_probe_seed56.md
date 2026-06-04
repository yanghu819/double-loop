# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0099
- train_sec: 310.7

## Clean Eval

- holes16: exact=0.9961, blank=0.9994, valid=0.9961
- holes20: exact=0.9922, blank=0.9981, valid=0.9980
- holes24: exact=0.9355, blank=0.9928, valid=0.9414
- holes28: exact=0.6953, blank=0.9637, valid=0.7109
- holes32: exact=0.2422, blank=0.8961, valid=0.2598

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
