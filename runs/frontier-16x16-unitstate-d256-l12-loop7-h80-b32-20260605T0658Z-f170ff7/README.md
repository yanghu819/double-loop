# 16x16 D256 Loop7 Harder-Hole Unit-State Probe

Purpose: test whether scaling hidden size and loop depth on the new unit-state mainline can move 16x16 from the h32/h48 foothold toward h64/h80.

Source SHA on GPU1: `f170ff74bff34392a3bb32dbe8a3a1eb16c7a4ae`

Run: `frontier-16x16-unitstate-d256-l12-loop7-h80-b32-20260605T0658Z-f170ff7`

Recorded UTC: `2026-06-05T07:43:59.260394+00:00`

## Setup

- GPU row: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Board: 16x16 Sudoku, 4x4 boxes, random holes
- Model: `D_MODEL=256`, `LAYERS=12`, `HEADS=16`, `HEAD_DIM=16`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=7`, final-loop CE
- Unit state: `UNIT_STATE_SCALE=1.0`, `UNIT_STATE_GATE_BIAS=-1.0`
- Unit loss: disabled, `UNIT_LOSS_WEIGHT=0`
- FutureSeed: `FUTURE_SEED_SCALE=1`
- Active rollout noise: disabled, `NOISE_SCALE=0`, `ROLLOUT_NOISE_SCALE=0`
- Kernel: `RWKV_KERNEL=cuda`
- Curriculum: `32-48:300,48-64:500,64-80:400`
- Eval holes: `48,64,80,96`, `FULL_EVAL_N=256`
- Batch: `FULL_BATCH=32`

## Training Trace

| step | stage | ce | loop1 loss | loop last loss | unit loss |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | 32-48 | 0.3058 | 0.2835 | 0.3058 | 0.0234 |
| 200 | 32-48 | 0.1498 | 0.1519 | 0.1498 | 0.0162 |
| 300 | 32-48 | 0.0953 | 0.1144 | 0.0953 | 0.0099 |
| 400 | 48-64 | 0.2047 | 0.2458 | 0.2047 | 0.0223 |
| 500 | 48-64 | 0.2111 | 0.2546 | 0.2111 | 0.0255 |
| 600 | 48-64 | 0.1851 | 0.2432 | 0.1851 | 0.0216 |
| 700 | 48-64 | 0.1724 | 0.2184 | 0.1724 | 0.0186 |
| 800 | 48-64 | 0.1781 | 0.2737 | 0.1781 | 0.0207 |
| 900 | 64-80 | 0.2966 | 0.4309 | 0.2966 | 0.0343 |
| 1000 | 64-80 | 0.2361 | 0.4235 | 0.2361 | 0.0272 |
| 1100 | 64-80 | 0.2193 | 0.4015 | 0.2193 | 0.0224 |
| 1200 | 64-80 | 0.1867 | 0.5612 | 0.1867 | 0.0188 |

Training took `2663.1s`.

## Loop Depth

Eval at 64 holes:

| loop | exact | valid | solved | blank acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.7713 |
| 3 | 0.0352 | 0.0352 | 0.0352 | 0.9437 |
| 5 | 0.0430 | 0.0430 | 0.0430 | 0.9432 |
| 7 | 0.0430 | 0.0430 | 0.0430 | 0.9432 |

## Hole Transfer

Exact solve rate by loop7:

| holes | exact | valid | solved | blank acc |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 0.4180 | 0.4219 | 0.4219 | 0.9766 |
| 64 | 0.0430 | 0.0430 | 0.0430 | 0.9432 |
| 80 | 0.0000 | 0.0000 | 0.0000 | 0.8724 |
| 96 | 0.0000 | 0.0000 | 0.0000 | 0.7820 |

## Insight

Scaling the unit-state mainline helps, but it does not open h80.

Compared with the earlier D192/L10/loop5 unit-state foothold, this run improves h48 exact from `0.2148` to `0.4180` and h64 exact from `0.0117` to `0.0430`. That is real signal: harder curriculum plus D256/L12/loop7 expands the 16x16 frontier.

The gain is not enough to justify direct width/depth escalation. Batch64 and batch48 both OOM on the 80GB GPU, while the successful batch32 run still ends with h80 exact `0.0` despite h80 blank accuracy `0.8724`. This is a global consistency failure, not local token recognition failure.

Loop depth remains valuable on hard holes. At 64 holes, loop1 exact is `0.0`, while loop3 already reaches `0.0352` and loop5/7 reach `0.0430`. During the 64-80 training stage, loop-last CE is much lower than loop1 CE, ending at `0.1867` versus `0.5612`. However, extra loops saturate after roughly loop5 at eval, so simply increasing `MAX_LOOPS` is unlikely to open h80.

## Decision

Do not launch 25x25 yet. The h80 failure means 25x25 would mostly amplify the current global consistency bottleneck.

Next high-ROI mechanism work should be structural:

1. Add persistent or true row/column/box unit tokens, not just pooled unit-state broadcast.
2. Add activation checkpointing only if it unlocks a structurally different run; direct D384/L16 is not justified.
3. If staying with pooled unit-state, test a longer h48-80 curriculum before more width, because h64 improved but h80 remained zero.

## Launch Notes

Three launch/resource aborts preceded the successful batch32 run:

- `frontier-16x16-unitstate-d256-l12-loop7-h80-20260605T0650Z-f170ff7`: selected project `.venv` lacked `torch`.
- `frontier-16x16-unitstate-d256-l12-loop7-h80-20260605T0653Z-f170ff7`: D256/L12/loop7 OOM at `FULL_BATCH=64`.
- `frontier-16x16-unitstate-d256-l12-loop7-h80-b48-20260605T0656Z-f170ff7`: D256/L12/loop7 OOM at `FULL_BATCH=48`.

No CPU smoke was used.

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
