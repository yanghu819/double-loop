# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 1.0
- train_ce: 0.3457
- train_sec: 267.2

## Clean Eval

- holes16: exact=0.0977, blank=0.8678, valid=0.0977
- holes20: exact=0.0176, blank=0.8083, valid=0.0176
- holes24: exact=0.0000, blank=0.7300, valid=0.0000
- holes28: exact=0.0000, blank=0.6585, valid=0.0000
- holes32: exact=0.0000, blank=0.5759, valid=0.0000

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
