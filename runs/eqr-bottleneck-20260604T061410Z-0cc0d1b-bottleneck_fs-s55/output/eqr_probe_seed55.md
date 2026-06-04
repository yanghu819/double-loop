# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 1.0
- train_ce: 0.3770
- train_sec: 275.8

## Clean Eval

- holes16: exact=0.1133, blank=0.8607, valid=0.1133
- holes20: exact=0.0039, blank=0.7840, valid=0.0039
- holes24: exact=0.0000, blank=0.7013, valid=0.0000
- holes28: exact=0.0000, blank=0.6198, valid=0.0000
- holes32: exact=0.0000, blank=0.5118, valid=0.0000

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
