# H108 Clean Frontier Test

Run: `h108frontier-12x12-d256-l12-loop5-s6200-20260611T1400Z-e446f4a`

Source SHA: `e446f4ae64b80067c40fce2e185f5b9de27fe3e9`

## Question

After h96 became supported by clean FutureSeed+loop scaling, does the same path open h108, or is h108 the next hard ceiling?

## Setup

- Board: 12x12, random holes, 3x4 boxes
- Model: D256/L12, 16 heads, head dim 16, loop5
- Forward dtype: bfloat16
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1400,84-96:2200,96-108:1600`
- Batch/eval: batch80, eval512
- Rollout: K1 only, loop readouts at 1/3/5
- Case bank: disabled to protect main result under the remaining lease
- Train time: `6591.7s`

## Results

| run | h84 exact | h96 exact | h108 exact | h120 exact | final CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| full-budget h96 | 0.8828 | 0.3164 | 0.0000 | not run | 0.1178 |
| h108 frontier | 0.9902 | 0.8848 | 0.1797 | 0.0000 | 0.1359 |

Loop readout at h108:

| loop | exact | blank_acc |
| ---: | ---: | ---: |
| 1 | 0.0000 | 0.3186 |
| 2 | 0.0000 | 0.5494 |
| 3 | 0.0020 | 0.7876 |
| 4 | 0.1328 | 0.8588 |
| 5 | 0.1797 | 0.8685 |

Key training checkpoints:

| step | stage | CE |
| ---: | --- | ---: |
| 2400 | 72-84 | 0.1655 |
| 4600 | 84-96 | 0.1679 |
| 4700 | 96-108 | 0.5078 |
| 5400 | 96-108 | 0.2327 |
| 6000 | 96-108 | 0.1560 |
| 6200 | 96-108 | 0.1359 |

## Decision

h108 is open under clean FutureSeed+loop scaling. This is not a selector win: K1 oracle and selectors are identical. It is also not a single-pass win: loop1 exact is `0.0`, while loop4 reaches `0.1328` and loop5 reaches `0.1797`.

The new closed boundary is h120. h120 blank accuracy is only `0.3439`, so this is not yet the same near-solved global-consistency regime as h96/h108. The next high-ROI question is whether h120 needs more clean curriculum/compute, a wider model, or a simple late-loop state revision. Do not go back to selector work.

No tag was created because the primary score is below the `0.50` strong-result threshold.
