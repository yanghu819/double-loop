# h120 Clean Continuation D256 Loop6 Step16300

Run: `h120resume-cleancont-d256-l12-loop6-s16300-20260615T0331Z-ddd9b3b`

Source SHA: `ddd9b3bd1fa29fb62a892b6eec8040c24f19fe61`

Recorded: `2026-06-15T04:25:43.815815+00:00`

## Hypothesis

The previous clean continuation opened h120 to `0.1777` after a late jump at step14300. This run asks whether the same D256/L12/loop6 model keeps turning more h120 hard-stage compute into more exact solves, or whether it starts filling more cells without improving full-board consistency.

This is a scaling readout, not an ablation. The mechanism is unchanged: fixed FutureSeed, no scratch, no feature noise, no selector, no repair.

## Configuration

- Board: 12x12, random holes, implicit 3x4 boxes.
- Resume point: `/huyang2/double-loop/.worktrees/h120resume-cont-d256-l12-loop6-s14300-20260615T0105Z-0e999dc/runs/h120resume-cont-d256-l12-loop6-s14300-20260615T0105Z-0e999dc/checkpoints/latest.pt`.
- Model: D256/L12, heads8, head_dim32, channel_mult4, loop6, `LOOP_LOSS=all`.
- FutureSeed: fixed update, decay `0.0`, no scratch state, no loop feedback, no loop-time conditioning.
- Noise: `NOISE_SCALE=0.0`, `SCRATCH_NOISE_SCALE=0.0`, `ROLLOUT_NOISE_SCALE=0.0`.
- Runtime: GPU1 only, CUDA visible device 0, bf16, RWKV statepassing CUDA kernel.
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:10500`.
- Checkpoint evals: steps `15300,16300`, holes `96,108,120,132`, eval_n `512`.

## Results

The run resumed cleanly from saved step14300, finished step16300, and wrote a new train checkpoint. `git_dirty=false`. Final train CE was `0.1815`, and continuation train time was `3173.3s`.

| point | h96 loop6 exact | h108 loop6 exact | h120 loop3 exact | h120 loop6 exact | h120 loop6 blank_acc | h132 loop6 exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step15300 checkpoint | 0.9961 | 0.9238 | 0.1328 | 0.1777 | 0.7526 | 0.0000 |
| step16300 checkpoint | 1.0000 | 0.9336 | 0.1074 | 0.1641 | 0.7527 | 0.0000 |
| final full eval | 0.9922 | 0.9453 | 0.1094 | 0.1758 | 0.7671 | 0.0000 |

Final h120 loop curve:

| loop | exact | valid_sudoku | blank_acc |
| --- | ---: | ---: | ---: |
| 1 | 0.0000 | 0.0000 | 0.4232 |
| 3 | 0.1094 | 0.1152 | 0.7496 |
| 6 | 0.1758 | 0.1836 | 0.7671 |

K1 oracle gap stayed zero at loop6.

## Insight

This run is the first clear low-marginal-return signal for the same-model h120 hard-stage continuation. More compute did increase local filling: h120 blank accuracy rose from the previous final `0.7465` to `0.7671`. But exact did not improve: previous final h120 loop6 exact was `0.1777`, step15300 was `0.1777`, step16300 checkpoint was `0.1641`, and final full eval was `0.1758`.

That means the current D256/L12/loop6 model is now better at making confident local assignments than at converting those assignments into globally consistent boards. This is not evidence against scaling in general. It is evidence against spending more GPU on the same model, same loop count, and same h120-only continuation.

Loop still matters. h120 loop1 exact is `0.0`, while loop6 reaches `0.1758`. The failure is not "loop useless"; it is that late loops no longer turn additional local accuracy into more complete solves.

h132 remains closed with blank accuracy only `0.1634`, so the frontier did not move beyond h120.

## Decision

Stop same-model h120 hard-stage continuation as a primary path. The next scaling experiment should change an actual scaling axis: larger effective model under resumable training, more efficient activation-checkpointed capacity, or a simple FutureSeed/loop state update that improves late correction. Do not spend the next budget on selector work, Sudoku repair, scratch-noise regularization, deeper loop repeats, or another identical h120 continuation.

No tag was created because the score is below `0.50`.
