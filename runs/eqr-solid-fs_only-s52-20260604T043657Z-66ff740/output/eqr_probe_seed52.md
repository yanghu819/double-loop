# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 4-8:200,8-12:400
- noise_mode: `none`
- future_seed_scale: 1.0
- train_ce: 0.0007
- train_sec: 217.9

## Clean Eval

- holes8: exact=1.0000, blank=1.0000, valid=1.0000
- holes12: exact=0.9961, blank=0.9997, valid=0.9961
- holes16: exact=0.9238, blank=0.9934, valid=0.9238
- holes20: exact=0.6719, blank=0.9715, valid=0.6719
- holes24: exact=0.2129, blank=0.9084, valid=0.2148

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
