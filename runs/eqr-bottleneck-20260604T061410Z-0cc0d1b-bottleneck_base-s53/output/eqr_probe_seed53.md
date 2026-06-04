# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.3203
- train_sec: 267.4

## Clean Eval

- holes16: exact=0.0938, blank=0.8607, valid=0.0938
- holes20: exact=0.0215, blank=0.8068, valid=0.0215
- holes24: exact=0.0000, blank=0.7410, valid=0.0000
- holes28: exact=0.0000, blank=0.6627, valid=0.0000
- holes32: exact=0.0000, blank=0.5797, valid=0.0000

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
