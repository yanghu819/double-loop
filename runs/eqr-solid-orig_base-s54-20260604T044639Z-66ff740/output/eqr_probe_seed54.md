# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 4-8:200,8-12:400
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0005
- train_sec: 208.4

## Clean Eval

- holes8: exact=1.0000, blank=1.0000, valid=1.0000
- holes12: exact=0.9883, blank=0.9984, valid=0.9883
- holes16: exact=0.9160, blank=0.9922, valid=0.9160
- holes20: exact=0.7129, blank=0.9709, valid=0.7129
- holes24: exact=0.2930, blank=0.9257, valid=0.2930

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
