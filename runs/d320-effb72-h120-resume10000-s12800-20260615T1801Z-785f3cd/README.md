# D320 Effective-Batch h120 Continuation to Step 12800

## Purpose

Test whether clean FutureSeed+loop scaling with a wider D320 backbone keeps improving h120 after the earlier step9800 result, or whether it plateaus below the D256 late-continuation frontier.

This is a single scaling curve, not an ablation sweep. It keeps the clean mainline fixed: no selector, no Sudoku repair, no scratch/noise mechanism.

## Setup

- GPU: GPU1 only
- Source SHA: `785f3cdca74aab1f6e9d6a4cc252a17e3aa63731`
- Resume checkpoint: previous D320 effective-batch `latest.pt` saved at step10000
- Board: 12x12 Sudoku, random holes
- Model: D320/L12/heads10/head_dim32, loop6, all-loop supervision
- Batch: microbatch 24, grad accumulation 3, effective batch 72
- Kernel/dtype: RWKV7 statepassing CUDA, bfloat16
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:7000`
- Eval holes: `96,108,120,132`

## Results

Checkpoint h120 curve:

| step | h120 loop3 exact | h120 loop4 exact | h120 loop5 exact | h120 loop6 exact | h120 loop6 blank_acc | train CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10800 | 0.0391 | 0.0801 | 0.0801 | 0.0820 | 0.6905 | 0.3762 |
| 11800 | 0.0410 | 0.0957 | 0.0996 | 0.1016 | 0.7077 | 0.3369 |
| 12800 | 0.0625 | 0.1074 | 0.1152 | 0.1191 | 0.7347 | 0.2818 |

Final full eval:

| holes | loop1 exact | loop3 exact | loop4 exact | loop5 exact | loop6 exact | loop6 blank_acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.0000 | 0.9902 | 0.9902 | 0.9902 | 0.9883 | 0.9993 |
| 108 | 0.0000 | 0.9238 | 0.9355 | 0.9355 | 0.9355 | 0.9938 |
| 120 | 0.0000 | 0.0664 | 0.1074 | 0.1191 | 0.1270 | 0.7331 |
| 132 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1642 |

Primary score: `0.126953125`.

## Interpretation

D320 effective-batch scaling keeps improving h120: the prior D320 step9800 checkpoint had h120 loop6 exact `0.0566`, and this run moves through `0.0820 -> 0.1016 -> 0.1191`, with final full eval `0.1270`.

The result does not beat the best D256 late-continuation score (`0.1777`). Width is therefore useful but not a shortcut. It needs enough training time to pay off, and the current D320 throughput is not ideal: the run used only about 27GB allocated / 32GB reserved, with moderate GPU utilization.

Loop remains necessary. At h120, loop1 exact is `0.0`, loop3 is `0.0664`, and loop6 is `0.1270`. Extra loop depth after loop4/5 still adds some exact solves here, unlike the earlier D256 plateau, but h132 remains fully closed.

## Decision

Continue the clean scaling direction, but do not blindly start a larger D384 table row. The next high-ROI options are:

1. Improve throughput/effective compute for D320, for example by testing a larger microbatch if memory allows.
2. Continue from the step12800 checkpoint only if buying a genuinely later point such as 13800/14800.
3. If the curve flattens, switch to a very simple FutureSeed/loop state update that improves late correction without Sudoku-specific priors.

Do not add selector, repair, unit-state, or scratch/noise work to this clean mainline.
