# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `feature_diff`
- future_seed_scale: 1.0
- train_ce: 0.0129
- train_sec: 555.1

## Clean Eval

- holes16: exact=0.9980, blank=0.9993, valid=1.0000
- holes20: exact=0.9863, blank=0.9977, valid=0.9883
- holes24: exact=0.9453, blank=0.9897, valid=0.9609
- holes28: exact=0.7773, blank=0.9657, valid=0.8105
- holes32: exact=0.4355, blank=0.9089, valid=0.4746

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
