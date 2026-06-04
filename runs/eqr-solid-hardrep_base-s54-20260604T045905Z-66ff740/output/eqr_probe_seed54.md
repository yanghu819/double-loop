# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0063
- train_sec: 310.9

## Clean Eval

- holes16: exact=0.9980, blank=0.9999, valid=0.9980
- holes20: exact=0.9922, blank=0.9987, valid=0.9941
- holes24: exact=0.9180, blank=0.9923, valid=0.9258
- holes28: exact=0.6797, blank=0.9606, valid=0.6953
- holes32: exact=0.3105, blank=0.8990, valid=0.3203

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
