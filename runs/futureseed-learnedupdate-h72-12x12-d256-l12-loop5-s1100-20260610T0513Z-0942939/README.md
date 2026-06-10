# Learned FutureSeed Update h72 Test

Run: `futureseed-learnedupdate-h72-12x12-d256-l12-loop5-s1100-20260610T0513Z-0942939`

Git SHA: `09429392f2a88ef09ac7b6fc2d0952e34397593f`

GPU: AIStation `GPU1`, A100 80GB, `CUDA_VISIBLE_DEVICES=0`

## Question

Fixed FutureSeed damping was too blunt. Can a learned update gate keep the benefit of damping while avoiding the h60 collapse?

## Setup

- Same clean 12x12 frontier recipe
- `FUTURE_SEED_DECAY=0.5`
- `FUTURE_SEED_UPDATE=learned`
- D256, L12, loop5, batch80
- Final-loop supervision
- h72 case bank: `eval_n=1024`, 8 requested cases per kind

## Results

Loop5 exact by holes:

| holes | exact | blank acc |
| ---: | ---: | ---: |
| 48 | 0.7005 | 0.9859 |
| 60 | 0.1927 | 0.9502 |
| 72 | 0.0000 | 0.8530 |
| 84 | 0.0000 | 0.6809 |

h72 case bank:

- Final exact: `0.0020`
- Final blank accuracy: `0.8553`
- Selected cases: 2 solved-by-loop, 8 almost-solved, 8 hard-failure
- Final wrong cells in selected almost/hard cases: 48
- Final wrong cells changed in the last shown loop: 3 / 48
- Stable-since histogram: loop1=16, loop2=20, loop3=9, loop5=3
- Mean final wrong-cell margin: `0.380`
- Mean final wrong-cell entropy: `0.802`

## Interpretation

This is a stronger negative result than fixed decay. The learned gate stayed near its initialization instead of recovering the clean baseline behavior, and both h60 and h72 collapsed.

The h72 stable-attractor problem is not solved by adding simple inertia to the FutureSeed state. In this implementation, damping mostly damages the representation before it can help later loops revise mistakes.

## Decision

Stop this damping/gating line. The next useful direction is not another decay value. The next clean test is pure hard-stage compute: keep the best clean mechanism and train longer on h60-72 to see whether the cliff is mostly training-budget limited.

