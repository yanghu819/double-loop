# D320 Effective-Batch h120 First-Checkpoint Scale Run

Run: `d320-effb72-h120-resume4000-s6800-20260615T0916Z-785f3cd`

Source SHA: `785f3cdca74aab1f6e9d6a4cc252a17e3aa63731`

Resume source: `d320-effb72-h120-step6800-r3-20260615T0513Z-785f3cd`, step4000 checkpoint.

## Hypothesis

The previous D320/B48 run was a weak test of width because lowering batch destroyed h96/h108 foundation. This run asks the fairer scaling question:

Can D320 keep effective batch 72 through gradient accumulation, preserve the h96/h108 foundation, and then open h120 without adding Sudoku repair, selector search, scratch state, or other task-specific tricks?

This is a bitter-lesson-aligned scale test: spend effort on general model capacity, effective batch, resumability, and training compute.

## Configuration

- Board: 12x12, random holes, 3x4 boxes.
- Model: D320/L12, heads10, head_dim32, channel_mult4.
- Loop: `MAX_LOOPS=6`, `LOOP_LOSS=all`.
- FutureSeed: fixed update, decay `0.0`.
- No scratch, no feature noise, no rollout noise, no selector, no repair.
- Batch: microbatch 24, `GRAD_ACCUM_STEPS=3`, effective batch 72.
- Runtime: GPU1 only, bf16, RWKV statepassing CUDA.
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:1000`.

## Results

Final train CE at step6800 was `0.5860`; train time after resume was `9251.6s`. Peak CUDA allocation/reservation was about `27104/27202 MB`.

| eval holes | loop6 exact | loop6 valid | loop6 blank_acc |
| --- | ---: | ---: | ---: |
| h96 | 0.9805 | 0.9824 | 0.9982 |
| h108 | 0.7344 | 0.7461 | 0.9694 |
| h120 | 0.0000 | 0.0000 | 0.5102 |
| h132 | 0.0000 | 0.0000 | 0.1635 |

h120 loop curve:

| loop | exact | valid | blank_acc |
| ---: | ---: | ---: | ---: |
| 1 | 0.0000 | 0.0000 | 0.3068 |
| 2 | 0.0000 | 0.0000 | 0.4235 |
| 3 | 0.0000 | 0.0000 | 0.4922 |
| 4 | 0.0000 | 0.0000 | 0.5065 |
| 5 | 0.0000 | 0.0000 | 0.5095 |
| 6 | 0.0000 | 0.0000 | 0.5102 |

K1 rollout oracle gap is zero.

## Insight

This is a mixed but high-information result.

D320 effective batch works as a scale axis. Compared with the previous D320/B48 first-checkpoint run, h96/h108 recover dramatically:

- D320/B48 at step6800: h96/h108/h120 exact `0.4336 / 0.0508 / 0.0000`.
- D320 effective batch72 at step6800: h96/h108/h120 exact `0.9805 / 0.7344 / 0.0000`.

So the earlier D320 failure was mostly a bad effective-compute test, not proof that width is useless.

But h120 still does not open. The h120 stage only had 1000 steps, final h120 CE was still high at `0.5860`, and blank accuracy was only `0.5102`. Loop improves local fill from `0.3068` to `0.5102`, but exact remains zero across all loops. This is not a selector problem and not a Sudoku-repair problem. It is not enough h120-stage learning/global consistency yet.

## Decision

Do not call D320 solved, and do not pivot to selector/repair. The next high-ROI scaling move is to resume this D320 effective-batch checkpoint and extend the h120 hard stage, with checkpoint evals every 1000-2000 steps. If h120 remains at zero after a much longer hard stage while blank accuracy rises, then the next bottleneck is state dynamics rather than capacity.

No tag was created because the primary h120 score is below `0.50`.
