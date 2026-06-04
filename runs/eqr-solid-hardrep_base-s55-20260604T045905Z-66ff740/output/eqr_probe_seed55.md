# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0103
- train_sec: 310.9

## Clean Eval

- holes16: exact=0.9980, blank=0.9993, valid=1.0000
- holes20: exact=0.9902, blank=0.9973, valid=0.9980
- holes24: exact=0.9590, blank=0.9942, valid=0.9688
- holes28: exact=0.7383, blank=0.9666, valid=0.7656
- holes32: exact=0.3164, blank=0.8998, valid=0.3320

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
