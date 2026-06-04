# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 4-8:200,8-12:400
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0008
- train_sec: 207.4

## Clean Eval

- holes8: exact=0.9980, blank=0.9998, valid=0.9980
- holes12: exact=0.9844, blank=0.9984, valid=0.9844
- holes16: exact=0.8984, blank=0.9905, valid=0.8984
- holes20: exact=0.6445, blank=0.9646, valid=0.6445
- holes24: exact=0.2207, blank=0.9089, valid=0.2207

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
