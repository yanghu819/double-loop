# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0054
- train_sec: 289.4

## Clean Eval

- holes16: exact=0.9941, blank=0.9990, valid=0.9941
- holes20: exact=0.9824, blank=0.9978, valid=0.9863
- holes24: exact=0.9141, blank=0.9891, valid=0.9258
- holes28: exact=0.7383, blank=0.9689, valid=0.7598
- holes32: exact=0.3281, blank=0.9046, valid=0.3340

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
