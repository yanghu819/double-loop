# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `feature_diff`
- future_seed_scale: 1.0
- train_ce: 0.0031
- train_sec: 476.5

## Clean Eval

- holes16: exact=1.0000, blank=1.0000, valid=1.0000
- holes20: exact=0.9766, blank=0.9969, valid=0.9824
- holes24: exact=0.9062, blank=0.9889, valid=0.9160
- holes28: exact=0.7090, blank=0.9697, valid=0.7188
- holes32: exact=0.2266, blank=0.8912, valid=0.2305

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
