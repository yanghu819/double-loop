# D320 Packed H120 Step22600 Continuation

- Timestamp: 2026-06-16T05:50Z launch, 2026-06-16T07:37Z completed
- Source SHA: `1adbda189b775d29c4ba2c30e31b7b92b5df556b`
- Remote worktree: `/huyang2/double-loop/.worktrees/d320-mb48eff96-h120-s22600-20260616T0550Z-1adbda1`
- Resume checkpoint: packed D320 step19600 `latest.pt`
- GPU: GPU1 only

## Hypothesis

The step19600 packed D320 run reached h120 exact `0.2891`, with h96/h108 still
strong and h132 still closed. This run tests whether the same clean training
trajectory continues to buy h120 solves after 20k steps, or whether the frontier
has entered a low-ROI local-fill plateau.

The mechanism is unchanged: fixed FutureSeed+loop, loop6, all-loop loss, CUDA
statepassing RWKV7, bf16, no selector, no repair, no scratch, no feature noise.

## Results

Checkpoint h120 loop6 exact:

- step20600: `0.2637`, blank_acc `0.8319`
- step21600: `0.2676`, blank_acc `0.8483`
- step22600: `0.2461`, blank_acc `0.8343`

Final full eval:

- score: `0.2754`
- h96/h108/h120/h132 loop6 exact: `0.9961` / `0.9492` / `0.2754` / `0.0000`
- h120 loop1/2/3/4/5/6 exact: `0.0000` / `0.0156` / `0.1914` / `0.2656` / `0.2715` / `0.2754`
- h120 loop6 blank_acc: `0.8397`
- K1 oracle gap: `0.0000`

## Decision

Do not continue the same packed D320 h120 hard-stage just because more steps are
available. The run keeps h96/h108 strong, but h120 final exact drops from the
previous `0.2891` to `0.2754`, and checkpoint exact only moves
`0.2637 -> 0.2676 -> 0.2461` while blank accuracy stays around `0.83-0.85`.
That is a low-ROI local-fill plateau, not evidence that another identical
3k-step continuation is the best next move.

Loop remains real and necessary: loop1 solves zero boards, loop3 reaches
`0.1914`, and loop4 does most of the remaining jump. But loop5 to loop6 adds
only `0.0039` exact, so a loop-count sweep is still not the right response.

The next high-information move should either improve effective scaling in a
general way or make a simple FutureSeed/loop state update more revisable in late
loops. Keep rejecting selector, Sudoku repair, and human-prior shortcuts unless
the evidence changes.
