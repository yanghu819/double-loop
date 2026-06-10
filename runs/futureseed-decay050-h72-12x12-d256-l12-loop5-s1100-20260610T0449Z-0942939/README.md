# Fixed FutureSeed Decay h72 Test

Run: `futureseed-decay050-h72-12x12-d256-l12-loop5-s1100-20260610T0449Z-0942939`

Git SHA: `09429392f2a88ef09ac7b6fc2d0952e34397593f`

GPU: AIStation `GPU1`, A100 80GB, `CUDA_VISIBLE_DEVICES=0`

## Question

Can a simple FutureSeed damping rule reduce early wrong-state lock-in at h72?

## Setup

This keeps the clean 12x12 frontier recipe and changes only FutureSeed update inertia:

- D256, L12, loop5, batch80
- Curriculum: `16-36:300,36-60:500,60-72:300`
- `FUTURE_SEED_DECAY=0.5`
- `FUTURE_SEED_UPDATE=fixed`
- Final-loop supervision
- h72 case bank: `eval_n=1024`, 8 cases per kind

## Results

Loop5 exact by holes:

| holes | exact | blank acc |
| ---: | ---: | ---: |
| 48 | 0.9062 | 0.9934 |
| 60 | 0.5052 | 0.9739 |
| 72 | 0.0521 | 0.9007 |
| 84 | 0.0000 | 0.7255 |

h72 case bank:

- Final exact: `0.0635`
- Final blank accuracy: `0.9007`
- Final wrong cells in selected almost/hard cases: 48
- Final wrong cells changed in the last shown loop: 2 / 48
- Stable-since histogram: loop1=17, loop2=21, loop3=8, loop5=2
- Mean final wrong-cell margin: `0.501`
- Mean final wrong-cell entropy: `0.671`

## Interpretation

Fixed damping is too blunt. It reduces the speed and final quality of the normal h60 regime, but it does not lift h72. The final wrong cells are still stable, and the higher mean margin suggests the model can remain confidently wrong even with smoothed FutureSeed state.

## Decision

Do not pursue fixed FutureSeed decay as a table. The only remaining simple damping idea worth one direct test is a learned update gate, where the model can choose the damping strength instead of receiving a fixed value.

