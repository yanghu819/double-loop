# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 4-8:200,8-12:400
- noise_mode: `feature_diff`
- future_seed_scale: 0.0
- train_ce: 0.0038
- train_sec: 374.2

## Clean Eval

- holes8: exact=1.0000, blank=1.0000, valid=1.0000
- holes12: exact=0.9863, blank=0.9985, valid=0.9863
- holes16: exact=0.9180, blank=0.9901, valid=0.9180
- holes20: exact=0.6309, blank=0.9610, valid=0.6309
- holes24: exact=0.2207, blank=0.9028, valid=0.2227

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
