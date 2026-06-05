# 25x25 Unit-Memory Conservative Feasibility Probe

Purpose: test whether the persistent unit-memory FutureSeed loop can form complete 25x25 Sudoku boards at low and moderate random-hole counts.

Source SHA on GPU1: `710bb46fbcf2c08b4dff1edf84c0e6baf6402ab4`

Run: `frontier-25x25-unitmem-feas-d160-l8-loop5-b32-20260605T0937Z-710bb46`

Recorded UTC: `2026-06-05T09:54:36.565612+00:00`

## Setup

- GPU row: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Board: 25x25 Sudoku, 5x5 boxes, random holes
- Model: `D_MODEL=160`, `LAYERS=8`, `HEADS=10`, `HEAD_DIM=16`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=5`, final-loop CE
- Unit memory: `UNIT_STATE_MODE=memory`, `UNIT_STATE_MEMORY_DECAY=0.7`, `UNIT_STATE_TOKEN_SCALE=1.0`
- Unit loss: disabled, `UNIT_LOSS_WEIGHT=0`
- FutureSeed: `FUTURE_SEED_SCALE=1`
- Active rollout noise: disabled, `NOISE_SCALE=0`, `ROLLOUT_NOISE_SCALE=0`
- Kernel: `RWKV_KERNEL=cuda`
- Curriculum: `25-50:300,50-75:500`
- Eval holes: `50,75,100,125`, `FULL_EVAL_N=256`
- Batch: `FULL_BATCH=32`

## Training Trace

| step | stage | ce | loop1 loss | loop last loss | unit loss |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | 25-50 | 1.5476 | 1.5189 | 1.5476 | 0.0501 |
| 200 | 25-50 | 0.0668 | 0.0681 | 0.0668 | 0.0031 |
| 300 | 25-50 | 0.0326 | 0.0467 | 0.0326 | 0.0016 |
| 400 | 50-75 | 0.0615 | 0.0780 | 0.0615 | 0.0043 |
| 500 | 50-75 | 0.0564 | 0.0683 | 0.0564 | 0.0045 |
| 600 | 50-75 | 0.0464 | 0.0669 | 0.0464 | 0.0036 |
| 700 | 50-75 | 0.0403 | 0.0539 | 0.0403 | 0.0031 |
| 800 | 50-75 | 0.0444 | 0.0601 | 0.0444 | 0.0034 |

Training took `997.3s`.

## Loop Depth

Eval at 75 holes:

| loop | exact | valid | solved | blank acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0117 | 0.0117 | 0.0117 | 0.9370 |
| 3 | 0.0586 | 0.0586 | 0.0586 | 0.9608 |
| 5 | 0.0742 | 0.0742 | 0.0742 | 0.9611 |

## Hole Transfer

Exact solve rate by loop5:

| holes | exact | valid | solved | blank acc |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 0.5156 | 0.5156 | 0.5156 | 0.9867 |
| 75 | 0.0742 | 0.0742 | 0.0742 | 0.9611 |
| 100 | 0.0000 | 0.0000 | 0.0000 | 0.9224 |
| 125 | 0.0000 | 0.0000 | 0.0000 | 0.8703 |

## Insight

25x25 is feasible under the unit-memory mainline. This run reaches h50 exact `0.5156` and h75 exact `0.0742` on a 625-cell board with a smaller D160/L8/loop5 model.

The initial step100 CE was high (`1.5476`), but optimization rapidly recovered by step200 (`0.0668`). This means the 25x25 issue is not raw CUDA throughput or basic optimization collapse under low/mid holes.

The frontier is now between h75 and h100. h100/h125 exact remain `0.0` despite h100 blank accuracy `0.9224`, so the same pattern holds: local blank recognition is strong, but full-board global consistency fails at higher sparsity.

Loop depth remains useful at 25x25: h75 exact rises from loop1 `0.0117` to loop5 `0.0742`.

## Decision

Treat 25x25 as supported at low holes and feasible at moderate holes under persistent unit memory. Do not call h100 supported yet.

Next high-ROI experiment should target the h75-h100 transition, for example `50-75:300,75-100:700` with eval h75/h90/h100/h110. Directly jumping to larger boards or high-hole 25x25 is lower ROI until h100 is nonzero.

## Artifacts

- `config.json`
- `score.json`
- `metadata.json`
- `logs/run.log`
- `logs/launcher.outer.log`
- `output/futureseed_loop_seed52.json`
- `output/futureseed_loop_seed52.md`
- `output/futureseed_loop_case_seed52.html`
- `source_HEAD.txt`
- `source.patch`
- `source_snapshot.tar.gz.sha256`
- `source_snapshot.ls.txt`
