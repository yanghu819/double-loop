# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0063
- train_sec: 322.6

## Clean Eval

- holes16: exact=0.9961, blank=0.9993, valid=0.9961
- holes20: exact=0.9727, blank=0.9964, valid=0.9746
- holes24: exact=0.8770, blank=0.9840, valid=0.8848
- holes28: exact=0.6680, blank=0.9563, valid=0.6875
- holes32: exact=0.2285, blank=0.8807, valid=0.2383

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
