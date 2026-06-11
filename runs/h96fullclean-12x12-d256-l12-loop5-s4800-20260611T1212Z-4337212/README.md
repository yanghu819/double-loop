# Full-Budget Clean H96 Scale Test

Run: `h96fullclean-12x12-d256-l12-loop5-s4800-20260611T1212Z-4337212`

Source SHA: `43372120a550473e7df4f5b45ff146bd72fefed0`

## Question

Does the clean FutureSeed+loop path still scale on 12x12 h96 if we preserve the full middle curriculum and spend a larger budget in the 84-96 hard stage, or was the previous h96 opening already near the mechanism ceiling?

## Setup

- Board: 12x12, random holes, 3x4 boxes
- Model: D256/L12, 16 heads, head dim 16, loop5
- Forward dtype: bfloat16
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1400,84-96:2400`
- Batch/eval: batch80, eval512
- Rollout: K1 only, loop readouts at 1/3/5
- Case bank: disabled to spend budget on the full run
- Train time: `5130.8s`

## Results

| run | h72 exact | h84 exact | h96 exact | h108 exact | h96 blank_acc | final CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean h96 baseline, 3400 steps | 0.8750 | 0.4740 | 0.0104 | 0.0000 | 0.7769 | 0.3931 |
| compressed h96 focus, 2400 steps | 0.9010 | 0.3828 | 0.0000 | 0.0000 | 0.7545 | 0.3850 |
| full-budget clean h96, 4800 steps | 0.9844 | 0.8828 | 0.3164 | 0.0000 | 0.9264 | 0.1178 |

Loop readout at h96:

| loop | exact | blank_acc |
| ---: | ---: | ---: |
| 1 | 0.0000 | 0.4001 |
| 2 | 0.0000 | 0.6760 |
| 3 | 0.0293 | 0.8703 |
| 4 | 0.2715 | 0.9215 |
| 5 | 0.3164 | 0.9264 |

Key training checkpoints:

| step | stage | CE |
| ---: | --- | ---: |
| 1000 | 60-72 | 0.1157 |
| 2400 | 72-84 | 0.1655 |
| 3200 | 84-96 | 0.3856 |
| 4000 | 84-96 | 0.3226 |
| 4500 | 84-96 | 0.2721 |
| 4800 | 84-96 | 0.1178 |

## Decision

This is strong positive evidence for clean compute/curriculum scaling. The h96 boundary moved from barely open (`0.0104`) to usable (`0.3164`) without adding Sudoku-specific repair rules or a new selector. The compressed h96 run was negative because it removed too much middle-stage foundation, not because h96 hard-stage exposure is useless.

Loop remains essential: h96 exact is `0.0` at loop1, `0.0293` at loop3, and `0.3164` at loop5. This supports FutureSeed+loop as the current mainline.

h108 is still closed with blank accuracy `0.5904`, so the next frontier is not selector work. It is either more clean scaling toward h108 or a simple late-loop state revision mechanism that improves global consistency without Sudoku-specific priors.

No tag was created because the primary score is below the `0.50` strong-result threshold.
