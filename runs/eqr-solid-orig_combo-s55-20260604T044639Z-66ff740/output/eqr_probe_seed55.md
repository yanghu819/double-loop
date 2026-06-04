# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 4-8:200,8-12:400
- noise_mode: `feature_diff`
- future_seed_scale: 1.0
- train_ce: 0.0029
- train_sec: 413.8

## Clean Eval

- holes8: exact=1.0000, blank=1.0000, valid=1.0000
- holes12: exact=0.9863, blank=0.9979, valid=0.9863
- holes16: exact=0.9180, blank=0.9917, valid=0.9180
- holes20: exact=0.6211, blank=0.9622, valid=0.6250
- holes24: exact=0.2891, blank=0.9129, valid=0.2910

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
