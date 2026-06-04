# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 1.0
- train_ce: 0.0084
- train_sec: 289.2

## Clean Eval

- holes16: exact=0.9980, blank=0.9999, valid=0.9980
- holes20: exact=0.9883, blank=0.9979, valid=0.9941
- holes24: exact=0.9336, blank=0.9921, valid=0.9355
- holes28: exact=0.7598, blank=0.9729, valid=0.7715
- holes32: exact=0.3770, blank=0.9177, valid=0.3828

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
