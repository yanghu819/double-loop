# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 4-8:200,8-12:400
- noise_mode: `feature_diff`
- future_seed_scale: 1.0
- train_ce: 0.0005
- train_sec: 375.2

## Clean Eval

- holes8: exact=1.0000, blank=1.0000, valid=1.0000
- holes12: exact=0.9844, blank=0.9976, valid=0.9844
- holes16: exact=0.9004, blank=0.9897, valid=0.9004
- holes20: exact=0.6152, blank=0.9614, valid=0.6172
- holes24: exact=0.2266, blank=0.8940, valid=0.2285

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
