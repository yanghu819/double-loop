# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `feature_diff`
- future_seed_scale: 1.0
- train_ce: 0.0173
- train_sec: 561.4

## Clean Eval

- holes16: exact=0.9980, blank=0.9993, valid=1.0000
- holes20: exact=0.9883, blank=0.9978, valid=0.9941
- holes24: exact=0.9297, blank=0.9906, valid=0.9395
- holes28: exact=0.7148, blank=0.9628, valid=0.7305
- holes32: exact=0.3340, blank=0.9063, valid=0.3457

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
