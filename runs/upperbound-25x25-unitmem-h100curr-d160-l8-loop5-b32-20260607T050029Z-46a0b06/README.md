# 25x25 Upper-Bound Probe: Unit Memory h100 Curriculum

## Purpose

This is an upper-bound probe for the current FutureSeed + recurrent loop + persistent unit-memory Sudoku paradigm on one 80GB A100. The experiment asks whether the current mechanism can move 25x25 Sudoku from low-hole feasibility into the h100 global-consistency regime, or whether the next gain requires a structural communication change.

## Configuration

- GPU row: GPU1 only
- Git SHA: `46a0b061c4a3fd02b9f378792283915ba3746026`
- Board: `25x25`, inferred `5x5` boxes
- Model: `D_MODEL=160`, `LAYERS=8`, `HEADS=10`, `HEAD_DIM=16`, `MAX_LOOPS=5`
- Mechanism: `FUTURE_SEED_SCALE=1`, `UNIT_STATE_MODE=memory`, `UNIT_STATE_MEMORY_DECAY=0.7`, `UNIT_STATE_SCALE=1.0`
- Curriculum: `50-75:300,75-100:700`
- Eval holes: `75,90,100,110`
- Batch/eval: `FULL_BATCH=32`, `FULL_EVAL_N=256`
- Kernel: `RWKV_KERNEL=cuda`

## Readout

| holes | loop5 exact | loop5 blank acc | interpretation |
| --- | ---: | ---: | --- |
| 75 | 0.1289 | 0.9714 | stronger than prior h75 feasibility |
| 90 | 0.0078 | 0.9510 | frontier starts around h90 |
| 100 | 0.0078 | 0.9381 | h100 is barely opened, not solved |
| 110 | 0.0000 | 0.9230 | closed |

Loop readout at h100:

| loop | exact | blank acc |
| --- | ---: | ---: |
| 1 | 0.0000 | 0.8880 |
| 2 | 0.0078 | 0.9351 |
| 3 | 0.0078 | 0.9377 |
| 4 | 0.0078 | 0.9379 |
| 5 | 0.0078 | 0.9381 |

Training was not the blocker:

- Step100 CE: `1.8481`
- Step200 CE: `0.0970`
- Step300 CE: `0.0503`
- Step1000 CE: `0.0715`
- Train time: `1216.8s`
- Peak observed memory: about `53.9GB`

## Insight

The current mechanism can optimize 25x25 h100 and even produces nonzero exact solve, but the score is far below the threshold where blind width/depth scaling is justified. Blank accuracy is high while exact is near zero, so the dominant failure is global consistency across row/column/box constraints rather than local digit recognition.

Loop matters, but only early: loop1 exact is zero, loop2 opens exact, and loops 3-5 mostly refine blank accuracy without changing exact. This says the present loop has some global correction capacity, but its communication substrate saturates quickly.

The better lesson signal is clear: more generic capacity is not the next highest-ROI step. The next mechanism should increase the amount of structured state that participates in the recurrent computation, especially true row/column/box unit tokens inside the RWKV sequence or another explicit constraint-state pathway.

## Decision

Continue the 25x25 direction, but do not run a blind model-size sweep. The next experiment should be a mechanism test:

1. Implement true unit tokens inside the recurrent sequence.
2. Compare against this run on h75/h90/h100 with the same budget.
3. Only then consider larger hidden dimension or deeper loops if h100 exact moves meaningfully.

No experiment tag was created because the primary h100 score is `0.0078`, far below the `0.50` tag threshold.
