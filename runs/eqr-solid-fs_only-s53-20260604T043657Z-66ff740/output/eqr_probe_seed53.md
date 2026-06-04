# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 4-8:200,8-12:400
- noise_mode: `none`
- future_seed_scale: 1.0
- train_ce: 0.0040
- train_sec: 212.8

## Clean Eval

- holes8: exact=1.0000, blank=1.0000, valid=1.0000
- holes12: exact=0.9746, blank=0.9980, valid=0.9746
- holes16: exact=0.8574, blank=0.9865, valid=0.8574
- holes20: exact=0.4863, blank=0.9513, valid=0.4863
- holes24: exact=0.1602, blank=0.8872, valid=0.1602

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
