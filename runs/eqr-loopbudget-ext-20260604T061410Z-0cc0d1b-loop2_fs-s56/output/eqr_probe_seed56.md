# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 1.0
- train_ce: 0.0079
- train_sec: 322.4

## Clean Eval

- holes16: exact=1.0000, blank=1.0000, valid=1.0000
- holes20: exact=0.9805, blank=0.9970, valid=0.9863
- holes24: exact=0.9199, blank=0.9903, valid=0.9277
- holes28: exact=0.6758, blank=0.9583, valid=0.6934
- holes32: exact=0.2578, blank=0.8981, valid=0.2734

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
