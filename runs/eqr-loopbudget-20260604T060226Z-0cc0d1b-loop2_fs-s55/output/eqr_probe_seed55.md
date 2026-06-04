# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 1.0
- train_ce: 0.0138
- train_sec: 289.7

## Clean Eval

- holes16: exact=0.9961, blank=0.9996, valid=0.9961
- holes20: exact=0.9785, blank=0.9970, valid=0.9844
- holes24: exact=0.9316, blank=0.9922, valid=0.9414
- holes28: exact=0.7129, blank=0.9630, valid=0.7324
- holes32: exact=0.2988, blank=0.9072, valid=0.3086

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
