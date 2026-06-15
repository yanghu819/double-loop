# D320 Packed H120 Launch Abort

- Timestamp: 2026-06-15T20:53:41Z launch, 2026-06-15T21:03:32Z abort
- Source SHA: `785f3cdca74aab1f6e9d6a4cc252a17e3aa63731`
- Remote worktree: `/huyang2/double-loop/.worktrees/d320-mb48eff96-h120-s13600-20260615T205559Z-785f3cd`
- GPU: GPU1 only

## What Happened

The first packed D320 launch failed before useful GPU training because the new
detached worktree could not compile/load the RWKV CUDA extension:

`RuntimeError: Ninja is required to load C++ extensions`

No model score should be read from this run.

## Lesson

Detached worktrees do not automatically inherit the repo-local `.cache/bin/ninja`
setup. Any new GPU1 worktree that uses the RWKV CUDA extension must either link
`/huyang2/double-loop/.cache/bin/ninja` into its own `.cache/bin` or prepend the
base repo cache and `/opt/conda/bin` to `PATH` before training.

The retry run `d320-mb48eff96-h120-s13600-r2-20260615T210332Z-785f3cd` applied
that fix and completed.

