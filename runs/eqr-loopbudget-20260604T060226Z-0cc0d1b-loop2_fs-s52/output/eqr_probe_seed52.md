# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 1.0
- train_ce: 0.0076
- train_sec: 289.6

## Clean Eval

- holes16: exact=0.9961, blank=0.9995, valid=0.9961
- holes20: exact=0.9922, blank=0.9990, valid=0.9922
- holes24: exact=0.9102, blank=0.9894, valid=0.9180
- holes28: exact=0.7051, blank=0.9681, valid=0.7188
- holes32: exact=0.2852, blank=0.9059, valid=0.3008

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
