# D320 Learned Loop Update Gate Probe

## Question

The packed D320 h120 run plateaued after step19600/22600: local fill stayed high,
but full-board exact stopped improving. This run tests a single simple state
dynamics change:

Can a learned scalar loop update gate let later loops keep correcting the board,
without adding selector, repair, scratch state, noise, or Sudoku-specific priors?

## Run

- Run name: `d320-updategate-remap-s23600-20260616T1005Z-8e2e217`
- Recorded UTC: `2026-06-16T10:43:13.604521+00:00`
- Git SHA: `8e2e217834f3482e448a14b8d37f168b299f8796`
- Dirty source: `false`
- Remote run dir: `/huyang2/double-loop/.worktrees/d320-updategate-remap-s23600-20260616T1005Z-8e2e217/runs/d320-updategate-remap-s23600-20260616T1005Z-8e2e217`
- Resume checkpoint: packed D320 step22600 clean run
- Board: `12x12`, box `3x4`, random holes
- Model: `D_MODEL=320`, `LAYERS=12`, `HEADS=10`, `HEAD_DIM=32`, `MAX_LOOPS=6`
- Batch: microbatch `48`, grad accumulation `2`, effective batch `96`
- Kernel: RWKV7 CUDA statepassing, bf16 forward
- Mechanism change: `LOOP_UPDATE_MODE=learned_gate`, `LOOP_UPDATE_GATE_INIT=0.95`
- Non-goals: no selector, no repair, no scratch, no feature-noise study, no loop-count sweep

The full source snapshot is retained in the remote run directory. Git tracks the
snapshot listing and hash instead of the 143MB tarball:

`f87c1bd1f45bea07e00d53695348cbb1aa9dd006c3334b888a81ba92a346a86e  source_snapshot.tar.gz`

## Results

Primary score:

- final h120 loop6 exact: `0.333984`
- checkpoint step23600 h120 loop6 exact: `0.349609`
- previous clean packed D320 reference:
  - step19600 final h120 loop6 exact: `0.289062`
  - step22600 final h120 loop6 exact: `0.275391`
- delayed loop-credit rejected reference:
  - step23100 h120 loop6 exact: `0.160156`

Final h120 loop curve:

| loop | exact | blank acc | valid |
| --- | ---: | ---: | ---: |
| 1 | 0.000000 | 0.485449 | 0.000000 |
| 2 | 0.025391 | 0.750374 | 0.025391 |
| 3 | 0.273438 | 0.829801 | 0.275391 |
| 4 | 0.314453 | 0.847266 | 0.326172 |
| 5 | 0.328125 | 0.849495 | 0.339844 |
| 6 | 0.333984 | 0.850033 | 0.345703 |

Final transfer loop6:

| holes | exact | blank acc | valid |
| --- | ---: | ---: | ---: |
| 96 | 0.996094 | 0.999573 | 0.996094 |
| 108 | 0.955078 | 0.994882 | 0.955078 |
| 120 | 0.333984 | 0.850033 | 0.345703 |
| 132 | 0.000000 | 0.169596 | 0.000000 |

Checkpoint loop6:

| step | h96 | h108 | h120 | h132 |
| --- | ---: | ---: | ---: | ---: |
| 23100 | 0.998047 | 0.935547 | 0.275391 | 0.000000 |
| 23600 | 0.998047 | 0.935547 | 0.349609 | 0.000000 |

Learned gate values at final eval:

| loop | gate_l | gate_h |
| --- | ---: | ---: |
| 1 | 0.8936 | 0.7880 |
| 2 | 0.7794 | 0.7081 |
| 3 | 0.8112 | 0.6761 |
| 4 | 0.9753 | 0.9610 |
| 5 | 0.9843 | 0.9824 |
| 6 | 0.9837 | 0.9831 |

## Decision

This is a real positive signal for simple FutureSeed/loop state dynamics. It
beats the same-config train-longer plateau and avoids the delayed-loss collapse.
The gate also learned a sensible shape: early loops are more conservative, while
late loops are near full update.

The result does not open h132. Treat this as a mechanism that improves h120
late-loop correction, not as proof that the current scale has crossed the next
frontier.

## Next

Do not run selector, repair, Sudoku priors, feature-noise tables, or deeper-loop
sweeps from this result. The next high-ROI choices are:

1. Continue from this learned-gate checkpoint with clean hard-stage compute only
   if the budget can reach another real checkpoint.
2. Test one slightly richer but still simple update rule, such as per-channel or
   state-conditioned gate, only if it preserves the same bitter-lesson property:
   more scalable learned computation, not hand-built Sudoku logic.
3. If h132 is the target, scale effective compute or capacity rather than adding
   local repair. h132 remains a distribution/compute frontier.
