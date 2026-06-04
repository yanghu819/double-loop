# EqR No-Hydra FutureSeed Probe

Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.

- board: 9x9
- holes: 8-16:300,16-24:600
- noise_mode: `none`
- future_seed_scale: 1.0
- train_ce: 0.0140
- train_sec: 323.6

## Clean Eval

- holes16: exact=0.9961, blank=0.9994, valid=0.9961
- holes20: exact=0.9863, blank=0.9978, valid=0.9922
- holes24: exact=0.9121, blank=0.9882, valid=0.9219
- holes28: exact=0.6621, blank=0.9648, valid=0.6680
- holes32: exact=0.2734, blank=0.8951, valid=0.2793

## Decision

Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled.
