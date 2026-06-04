# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 1.0
- train_ce: 0.6523
- train_sec: 278.3

## Clean Eval

- holes16: exact=0.0000, blank=0.6842, valid=0.0000
- holes20: exact=0.0000, blank=0.6083, valid=0.0000
- holes24: exact=0.0000, blank=0.5241, valid=0.0000
- holes28: exact=0.0000, blank=0.4672, valid=0.0000
- holes32: exact=0.0000, blank=0.4050, valid=0.0000

## Decision

Pivot: exact remains near zero; inspect global consistency or reduce task difficulty.
