# D320 Packed H120 Continuation

- Timestamp: 2026-06-15T21:01:54Z launch, 2026-06-15T21:34:09Z recorded
- Source SHA: `785f3cdca74aab1f6e9d6a4cc252a17e3aa63731`
- Remote worktree: `/huyang2/double-loop/.worktrees/d320-mb48eff96-h120-s13600-r2-20260615T210332Z-785f3cd`
- Resume checkpoint: D320 effective-batch step12800 `latest.pt`
- GPU: GPU1 only

## Hypothesis

The previous D320 effective-batch continuation improved h120 but only used about
27GB allocated / 32GB reserved on an 80GB GPU. Before adding a new mechanism or a
larger width, test whether the same clean FutureSeed+loop path benefits from
better packing: microbatch `48`, accumulation `2`, effective batch `96`.

This is a scaling-system question, not an ablation table: if packing increases
GPU utilization and still moves h120, the next high-ROI axis is longer clean
packed compute. If h120 stalls, the next axis should change the state update
mechanism instead of repeating the same hard stage.

## Setup

- Board: 12x12 Sudoku, random holes, 3x4 boxes
- Model: D320/L12, heads10, head_dim32, loop6, all-loop loss
- Kernel: CUDA statepassing RWKV7, bf16
- Mainline: fixed FutureSeed+loop, no scratch, no noise, no selector, no repair
- Curriculum tail: h108-120 hard stage continued from step12800 to step13600
- Evaluation: checkpoint eval at steps 13200 and 13600 on h96/h108/h120/h132

## Results

Checkpoint h120 loop6 exact:

- step13200: `0.1289`, blank_acc `0.7398`
- step13600: `0.1777`, blank_acc `0.7628`

Final full eval:

- h96/h108/h120/h132 loop6 exact: `0.9941` / `0.9492` / `0.1738` / `0.0000`
- h120 loop1/3/4/5/6 exact: `0.0000` / `0.0820` / `0.1562` / `0.1699` / `0.1738`
- h120 loop6 blank_acc: `0.7611`
- K1 oracle gap: `0.0000`

Observed during training, GPU memory was about 60-64GB and utilization was near
100%, so microbatch `48` is a better 80GB packing point than the previous
microbatch `24` route.

## Decision

Continue the clean packed D320 route only if buying a genuinely later point than
step13600. This run nearly matches the D256 late-continuation best but does not
clearly beat it, so it is not a width breakthrough. The useful lesson is that
throughput/effective-batch scaling is real and should be exhausted before adding
Sudoku-specific fixes.

Do not spend on selector work: K1 oracle gap is zero. Do not spend on h132 yet:
h132 remains closed with exact `0.0` and blank_acc only about `0.165`.

