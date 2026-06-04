# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.5156
- train_sec: 267.4

## Clean Eval

- holes16: exact=0.0098, blank=0.7731, valid=0.0098
- holes20: exact=0.0020, blank=0.7052, valid=0.0020
- holes24: exact=0.0000, blank=0.6323, valid=0.0000
- holes28: exact=0.0000, blank=0.5543, valid=0.0000
- holes32: exact=0.0000, blank=0.4720, valid=0.0000

## Decision

Pivot: exact remains near zero; inspect global consistency or reduce task difficulty.
