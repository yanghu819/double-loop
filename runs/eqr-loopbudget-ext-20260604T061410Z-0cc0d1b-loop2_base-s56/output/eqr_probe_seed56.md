# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 0.0
- train_ce: 0.0055
- train_sec: 322.1

## Clean Eval

- holes16: exact=0.9941, blank=0.9991, valid=0.9941
- holes20: exact=0.9844, blank=0.9978, valid=0.9883
- holes24: exact=0.9141, blank=0.9913, valid=0.9160
- holes28: exact=0.6777, blank=0.9610, valid=0.6855
- holes32: exact=0.2656, blank=0.8961, valid=0.2754

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
