# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `feature_diff`
- future_seed_scale: 1.0
- train_ce: 0.0055
- train_sec: 557.2

## Clean Eval

- holes16: exact=1.0000, blank=1.0000, valid=1.0000
- holes20: exact=0.9727, blank=0.9965, valid=0.9785
- holes24: exact=0.8867, blank=0.9833, valid=0.9023
- holes28: exact=0.6289, blank=0.9570, valid=0.6406
- holes32: exact=0.2188, blank=0.8835, valid=0.2246

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
