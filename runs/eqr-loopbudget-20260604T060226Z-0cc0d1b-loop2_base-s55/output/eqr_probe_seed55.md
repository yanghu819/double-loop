# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0115
- train_sec: 289.4

## Clean Eval

- holes16: exact=0.9961, blank=0.9993, valid=0.9961
- holes20: exact=0.9844, blank=0.9976, valid=0.9863
- holes24: exact=0.9336, blank=0.9923, valid=0.9434
- holes28: exact=0.7148, blank=0.9668, valid=0.7383
- holes32: exact=0.3008, blank=0.9003, valid=0.3164

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
