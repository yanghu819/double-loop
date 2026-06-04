# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 4-8:200,8-12:400
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0112
- train_sec: 207.1

## Clean Eval

- holes8: exact=1.0000, blank=1.0000, valid=1.0000
- holes12: exact=0.9844, blank=0.9977, valid=0.9844
- holes16: exact=0.8828, blank=0.9874, valid=0.8828
- holes20: exact=0.5938, blank=0.9563, valid=0.5957
- holes24: exact=0.2266, blank=0.8911, valid=0.2285

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
