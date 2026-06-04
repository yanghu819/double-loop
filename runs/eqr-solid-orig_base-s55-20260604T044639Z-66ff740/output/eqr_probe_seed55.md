# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 4-8:200,8-12:400
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0003
- train_sec: 207.6

## Clean Eval

- holes8: exact=1.0000, blank=1.0000, valid=1.0000
- holes12: exact=0.9902, blank=0.9985, valid=0.9902
- holes16: exact=0.9219, blank=0.9933, valid=0.9219
- holes20: exact=0.5977, blank=0.9634, valid=0.5977
- holes24: exact=0.2012, blank=0.9089, valid=0.2012

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
