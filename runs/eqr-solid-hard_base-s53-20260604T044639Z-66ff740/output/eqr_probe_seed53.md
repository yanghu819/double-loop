# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0120
- train_sec: 274.8

## Clean Eval

- holes16: exact=0.9980, blank=0.9999, valid=0.9980
- holes20: exact=0.9883, blank=0.9976, valid=0.9941
- holes24: exact=0.9355, blank=0.9913, valid=0.9414
- holes28: exact=0.7500, blank=0.9701, valid=0.7656
- holes32: exact=0.3398, blank=0.9028, valid=0.3496

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
