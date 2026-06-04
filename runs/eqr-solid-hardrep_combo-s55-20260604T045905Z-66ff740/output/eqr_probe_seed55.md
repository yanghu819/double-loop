# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `feature_diff`
- future_seed_scale: 1.0
- train_ce: 0.0092
- train_sec: 557.8

## Clean Eval

- holes16: exact=0.9961, blank=0.9991, valid=0.9980
- holes20: exact=0.9844, blank=0.9972, valid=0.9883
- holes24: exact=0.9355, blank=0.9924, valid=0.9414
- holes28: exact=0.6758, blank=0.9619, valid=0.6992
- holes32: exact=0.2402, blank=0.8909, valid=0.2500

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
