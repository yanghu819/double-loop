# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 4-8:200,8-12:400
- noise_mode: `feature_diff`
- future_seed_scale: 0.0
- train_ce: 0.0006
- train_sec: 373.9

## Clean Eval

- holes8: exact=1.0000, blank=1.0000, valid=1.0000
- holes12: exact=0.9766, blank=0.9971, valid=0.9766
- holes16: exact=0.8672, blank=0.9879, valid=0.8672
- holes20: exact=0.6016, blank=0.9611, valid=0.6016
- holes24: exact=0.2070, blank=0.8975, valid=0.2070

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
