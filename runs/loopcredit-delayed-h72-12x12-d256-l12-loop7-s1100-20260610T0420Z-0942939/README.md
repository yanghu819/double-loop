# Loop-Credit Delayed h72 Test

Run: `loopcredit-delayed-h72-12x12-d256-l12-loop7-s1100-20260610T0420Z-0942939`

Git SHA: `09429392f2a88ef09ac7b6fc2d0952e34397593f`

GPU: AIStation `GPU1`, A100 80GB, `CUDA_VISIBLE_DEVICES=0`

## Question

Can more recurrent loops plus delayed loop supervision break the h72 stable wrong attractors?

## Setup

This was a single mechanism test, not an ablation table:

- Same 12x12 frontier recipe as the h72 diagnostics run
- Increased loop budget from 5 to 7
- Batch reduced from 80 to 64 to fit loop7 backprop
- `LOOP_LOSS=delayed`, `LOOP_LOSS_START=3`, `LOOP_LOSS_POWER=2.0`
- h72 case bank kept at `eval_n=1024`, 8 cases per kind

## Results

Loop7 exact by holes:

| holes | exact | blank acc |
| ---: | ---: | ---: |
| 48 | 0.8464 | 0.9913 |
| 60 | 0.4245 | 0.9676 |
| 72 | 0.0156 | 0.8793 |
| 84 | 0.0000 | 0.6939 |

Within h60:

| loop | exact | blank acc |
| ---: | ---: | ---: |
| 1 | 0.0000 | 0.8020 |
| 3 | 0.4036 | 0.9664 |
| 5 | 0.4167 | 0.9674 |
| 7 | 0.4245 | 0.9676 |

h72 case bank:

- Final exact: `0.0283`
- Final blank accuracy: `0.8860`
- Final wrong cells in selected almost/hard cases: 48
- Final wrong cells changed in the last shown loop: 0 / 48
- Stable-since histogram: loop1=17, loop2=18, loop3=9, loop5=4, loop7=0

## Interpretation

This is a clean negative result. Delaying supervision and adding two loops made optimization slower and reduced final accuracy. Extra loops still did not revise the stable final wrong cells.

The h72 issue is not that the model lacks two more refinement steps. It enters stable wrong states early, and simply supervising later loops does not teach it to escape them.

## Decision

Stop this direction. Do not expand it into a table over loop loss starts or loop counts.

Next high-ROI lever: make the FutureSeed/update path itself less eager to lock early states. A simple damping or learned update gate is more aligned with the observed failure than more loop-credit shaping.

