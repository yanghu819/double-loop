# H96 Focused Curriculum Probe

Run: `h96focus-12x12-d256-l12-loop5-s2400-20260611T1046Z-edb4804`

Source SHA: `edb4804c39938d72fc5c99e9994766c6cd8973b8`

## Question

Can the clean FutureSeed+loop 12x12 h96 boundary be opened by cutting early curriculum and spending most remaining budget directly on the 84-96 hard stage, or does h96 need a fuller middle-stage foundation?

## Setup

- Board: 12x12, random holes, 3x4 boxes
- Model: D256/L12, 16 heads, head dim 16, loop5
- Forward dtype: bfloat16
- Curriculum: `16-36:100,36-60:150,60-72:250,72-84:500,84-96:1400`
- Batch/eval: batch80, eval384
- Feature noise: existing `feature_diff` default
- Case bank: disabled to fit the remaining GPU1 lease

## Results

| run | h72 exact | h84 exact | h96 exact | h108 exact | h96 blank_acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean h96 full baseline | 0.8750 | 0.4740 | 0.0104 | 0.0000 | 0.7769 |
| h96-focused compressed | 0.9010 | 0.3828 | 0.0000 | 0.0000 | 0.7545 |

Training reached final CE `0.3850`, close to the full h96 baseline final CE `0.3931`, but h96 exact still collapsed to `0.0`.

Key training checkpoints:

| step | stage | CE |
| ---: | --- | ---: |
| 100 | 16-36 | 1.5464 |
| 300 | 60-72 | 1.6126 |
| 500 | 60-72 | 0.8148 |
| 1000 | 72-84 | 0.3012 |
| 1200 | 84-96 | 0.6266 |
| 1800 | 84-96 | 0.4073 |
| 2400 | 84-96 | 0.3850 |

Loop behavior at h96:

| loop | exact | blank_acc |
| ---: | ---: | ---: |
| 1 | 0.0000 | 0.4309 |
| 2 | 0.0000 | 0.6253 |
| 3 | 0.0000 | 0.7348 |
| 4 | 0.0000 | 0.7522 |
| 5 | 0.0000 | 0.7545 |

## Decision

This is negative evidence for compressed h96 schedules. Earlier hard-stage exposure improves h72 transfer, but it hurts h84 and loses the tiny h96 opening from the full baseline.

The useful lesson is that h96 is not unlocked by simply entering the hard stage earlier. The middle curriculum seems to build the global consistency foundation that the final hard stage depends on. Do not run more compressed h96 variants. Next useful work should be either a full-length clean h96 hard-stage run or a simple mechanism that changes late-loop state revision.
