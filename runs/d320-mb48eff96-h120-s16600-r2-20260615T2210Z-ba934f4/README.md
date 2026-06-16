# D320 Packed H120 Long Continuation

- Timestamp: 2026-06-15T22:10Z launch, 2026-06-15T23:58:03Z recorded
- Source SHA: `ba934f4f717dfb7ce987d3384b03147ba2df72c9`
- Remote worktree: `/huyang2/double-loop/.worktrees/d320-mb48eff96-h120-s16600-r2-20260615T2210Z-ba934f4`
- Resume checkpoint: packed D320 step13600 `latest.pt`
- GPU: GPU1 only

## Hypothesis

The step13600 packed D320 run nearly reached the previous D256 best. This run
tests whether that was a lucky eval point or the beginning of a new clean scaling
curve. The only changed axis is more packed hard-stage compute: same model,
same FutureSeed+loop mechanism, same microbatch `48`, same accumulation `2`.

Decision value:

- If h120 exact passes `0.20`, continue clean packed scaling.
- If blank accuracy rises while exact stalls near `0.17-0.18`, switch to a simple
  FutureSeed/loop state update instead of burning more identical hard-stage time.

## Setup

- Board: 12x12 Sudoku, random holes, 3x4 boxes
- Model: D320/L12, heads10, head_dim32, loop6, all-loop loss
- Kernel: CUDA statepassing RWKV7, bf16
- Mainline: fixed FutureSeed+loop, no scratch, no noise, no selector, no repair
- Curriculum tail: h108-120 hard stage continued from step13600 to step16600
- Evaluation: checkpoint eval at steps 14600, 15600, 16600 on h96/h108/h120/h132

## Results

Checkpoint h120 loop6 exact:

- step14600: `0.1445`, blank_acc `0.7803`
- step15600: `0.1738`, blank_acc `0.7709`
- step16600: `0.1953`, blank_acc `0.7889`

Final full eval:

- score: `0.2363`
- h96/h108/h120/h132 loop6 exact: `0.9941` / `0.9297` / `0.2363` / `0.0000`
- h120 loop1/3/4/5/6 exact: `0.0000` / `0.1621` / `0.2188` / `0.2363` / `0.2363`
- h120 loop6 blank_acc: `0.7964`
- K1 oracle gap: `0.0000`

## Decision

Continue clean packed D320 scaling. This run clears the old h120 plateau: D256
late continuation and step13600 packed D320 both sat around `0.17-0.18`, while
this run reaches `0.2363`. That is enough evidence that h120 is still compute
limited under the clean FutureSeed+loop path.

Do not add selector work: K1 oracle gap is still zero. Do not add Sudoku repair
or unit priors to this mainline: the useful move was more general compute and
better GPU packing. Do not simply increase loop count: loop5 and loop6 exact are
identical on the final eval, so the next clean scaling point should buy more
training, not more recurrent depth.

The next high-ROI experiment is a packed continuation from step16600 to a later
checkpoint, with fixed evals. Stop only if exact stalls while blank accuracy keeps
rising, because that would mean local filling has outpaced global consistency.

