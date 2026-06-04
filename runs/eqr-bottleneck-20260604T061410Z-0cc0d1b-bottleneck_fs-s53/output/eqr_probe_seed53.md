# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 1.0
- train_ce: 0.5625
- train_sec: 267.1

## Clean Eval

- holes16: exact=0.0020, blank=0.7059, valid=0.0020
- holes20: exact=0.0000, blank=0.6453, valid=0.0000
- holes24: exact=0.0000, blank=0.5725, valid=0.0000
- holes28: exact=0.0000, blank=0.5068, valid=0.0000
- holes32: exact=0.0000, blank=0.4413, valid=0.0000

## Decision

Pivot: exact remains near zero; inspect global consistency or reduce task difficulty.
