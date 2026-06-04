# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 4-8:200,8-12:400
- noise_mode: `feature_diff`
- future_seed_scale: 1.0
- train_ce: 0.0004
- train_sec: 375.2

## Clean Eval

- holes8: exact=0.9980, blank=0.9998, valid=0.9980
- holes12: exact=0.9844, blank=0.9982, valid=0.9844
- holes16: exact=0.9121, blank=0.9921, valid=0.9121
- holes20: exact=0.6777, blank=0.9698, valid=0.6777
- holes24: exact=0.2480, blank=0.9105, valid=0.2480

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
