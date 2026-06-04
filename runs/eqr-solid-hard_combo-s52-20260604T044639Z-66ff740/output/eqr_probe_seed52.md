# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `feature_diff`
- future_seed_scale: 1.0
- train_ce: 0.0074
- train_sec: 475.1

## Clean Eval

- holes16: exact=1.0000, blank=1.0000, valid=1.0000
- holes20: exact=0.9863, blank=0.9970, valid=0.9922
- holes24: exact=0.9082, blank=0.9908, valid=0.9180
- holes28: exact=0.7461, blank=0.9695, valid=0.7617
- holes32: exact=0.3086, blank=0.9048, valid=0.3223

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
