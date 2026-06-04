# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.3340
- train_sec: 267.5

## Clean Eval

- holes16: exact=0.1094, blank=0.8749, valid=0.1094
- holes20: exact=0.0098, blank=0.8088, valid=0.0098
- holes24: exact=0.0000, blank=0.7313, valid=0.0000
- holes28: exact=0.0000, blank=0.6526, valid=0.0000
- holes32: exact=0.0000, blank=0.5673, valid=0.0000

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
