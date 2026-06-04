# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 4-8:200,8-12:400
- noise_mode: `feature_diff`
- future_seed_scale: 1.0
- train_ce: 0.0002
- train_sec: 412.1

## Clean Eval

- holes8: exact=0.9980, blank=0.9995, valid=0.9980
- holes12: exact=0.9883, blank=0.9980, valid=0.9883
- holes16: exact=0.8945, blank=0.9880, valid=0.8945
- holes20: exact=0.6543, blank=0.9666, valid=0.6543
- holes24: exact=0.2305, blank=0.9102, valid=0.2344

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
