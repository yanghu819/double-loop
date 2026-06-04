# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0067
- train_sec: 311.0

## Clean Eval

- holes16: exact=0.9961, blank=0.9990, valid=0.9961
- holes20: exact=0.9883, blank=0.9981, valid=0.9902
- holes24: exact=0.9062, blank=0.9872, valid=0.9199
- holes28: exact=0.6934, blank=0.9645, valid=0.7051
- holes32: exact=0.2539, blank=0.8927, valid=0.2559

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
