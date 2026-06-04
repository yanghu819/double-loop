# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0061
- train_sec: 279.8

## Clean Eval

- holes16: exact=1.0000, blank=1.0000, valid=1.0000
- holes20: exact=0.9805, blank=0.9973, valid=0.9844
- holes24: exact=0.9082, blank=0.9893, valid=0.9199
- holes28: exact=0.6777, blank=0.9673, valid=0.6914
- holes32: exact=0.2188, blank=0.8876, valid=0.2246

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
