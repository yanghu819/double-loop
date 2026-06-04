# 12x12 FutureSeed RWKV Frontier Probe

Purpose: test whether the current FutureSeed RWKV recurrent-loop paradigm can move beyond 9x9 Sudoku into a larger real constraint system.

Source SHA on GPU1: `f67e6a2e909dc39ce5ba48f622d93b1ec43f5ab3`

Run: `frontier-12x12-d256-h72-20260604T0920Z-f67e6a2`

Recorded UTC: `2026-06-04T09:55:59.532238+00:00`

## Setup

- GPU row: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Board: 12x12 Sudoku, 3x4 boxes, random holes
- Model: `D_MODEL=256`, `LAYERS=12`, `HEADS=16`, `HEAD_DIM=16`, `CHANNEL_MULT=4`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- FutureSeed: `FUTURE_SEED_SCALE=1`
- Active noise injection: none, `NOISE_SCALE=0`, `ROLLOUT_NOISE_SCALE=0`
- Kernel: `RWKV_KERNEL=cuda`
- Curriculum: `16-36:300,36-60:500,60-72:300`
- Eval holes: `48,60,72,84`, `FULL_EVAL_N=384`

## Training Trace

| step | stage | ce | loop1 loss | loop last loss |
| ---: | --- | ---: | ---: | ---: |
| 100 | 16-36 | 1.5160 | 1.5117 | 1.5160 |
| 200 | 16-36 | 1.2282 | 1.2346 | 1.2282 |
| 300 | 16-36 | 0.8291 | 0.8669 | 0.8291 |
| 400 | 36-60 | 0.8054 | 0.9941 | 0.8054 |
| 500 | 36-60 | 0.2922 | 0.6837 | 0.2922 |
| 600 | 36-60 | 0.1669 | 0.6449 | 0.1669 |
| 700 | 36-60 | 0.1258 | 0.6781 | 0.1258 |
| 800 | 36-60 | 0.0756 | 0.6443 | 0.0756 |
| 900 | 60-72 | 0.2395 | 1.0061 | 0.2395 |
| 1000 | 60-72 | 0.1941 | 1.0132 | 0.1941 |
| 1100 | 60-72 | 0.1857 | 1.0238 | 0.1857 |

Training took `2111.1s`.

## Hole Transfer

Exact solve rate by loop:

| holes | loop1 | loop2 | loop3 | loop4 | loop5 | loop5-loop1 | loop5 blank acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48 | 0.0000 | 0.6849 | 0.8776 | 0.8828 | 0.8880 | +0.8880 | 0.9944 |
| 60 | 0.0000 | 0.1615 | 0.4427 | 0.4844 | 0.5052 | +0.5052 | 0.9743 |
| 72 | 0.0000 | 0.0026 | 0.0234 | 0.0391 | 0.0443 | +0.0443 | 0.8881 |
| 84 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | 0.6993 |

## Insight

12x12 is inside the current paradigm's support, but only up to a clear hard-hole frontier. The model solves h60 at `0.5052` exact and h48 at `0.8880`, while h72 is only a thin nonzero fringe and h84 collapses.

Loop depth is not optional at this scale. Loop1 exact is `0.0` for every evaluated hole count, but loop5 reaches `0.5052` on h60. The mechanism is recurrent global refinement after FutureSeed, not one-shot local prediction.

The h72 and h84 failures are board-level consistency failures. h72 still has `0.8881` blank accuracy but only `0.0443` exact, so local cell accuracy is far ahead of complete-board consistency.

## Decision

Continue the scale-frontier search to 16x16, but only as a smaller viability probe. 12x12 D256/L12 already used nearly all GPU1 memory, so 16x16 should reduce batch/model rather than blindly copying the 12x12 capacity.

## Provenance Note

The run was clean at launch: `git_dirty=false`. `source_HEAD.txt`, `source.patch`, and `source_snapshot.tar.gz` are archived on the GPU1/local run directories.

The source snapshot is large because prior tracked source snapshots are already part of `HEAD`. The tarball is `139.91MB`, which exceeds GitHub's `100MB` hard file limit, so GitHub tracks its `sha256` and `ls` metadata instead of the tarball blob. This is a tracking-system lesson: source snapshots must move out of the Git tree for larger experiments.

## Artifacts

- `config.json`
- `score.json`
- `metadata.json`
- `logs/run.log`
- `logs/launcher.log`
- `launch.env`
- `output/futureseed_loop_seed52.json`
- `output/futureseed_loop_seed52.md`
- `output/futureseed_loop_case_seed52.html`
- `source_HEAD.txt`
- `source.patch`
- `source_snapshot.tar.gz` on GPU1/local disk
- `source_snapshot.tar.gz.sha256`
- `source_snapshot.ls.txt`
