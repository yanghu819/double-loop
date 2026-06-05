# 16x16 Unit-Memory H80 Curriculum Probe

Purpose: test whether the h80 failure in the first persistent unit-memory run was a hard-stage curriculum length issue or a remaining architecture boundary.

Source SHA on GPU1: `0be64f46c954cdb578ebb34b0c6bcd63b4c64b32`

Run: `frontier-16x16-unitmem-h80curr-d192-l10-loop7-b48-20260605T0850Z-0be64f4`

Recorded UTC: `2026-06-05T09:28:47.715789+00:00`

## Setup

- GPU row: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Board: 16x16 Sudoku, 4x4 boxes, random holes
- Model: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=7`, final-loop CE
- Unit memory: `UNIT_STATE_MODE=memory`, `UNIT_STATE_MEMORY_DECAY=0.7`, `UNIT_STATE_TOKEN_SCALE=1.0`
- Unit loss: disabled, `UNIT_LOSS_WEIGHT=0`
- FutureSeed: `FUTURE_SEED_SCALE=1`
- Active rollout noise: disabled, `NOISE_SCALE=0`, `ROLLOUT_NOISE_SCALE=0`
- Kernel: `RWKV_KERNEL=cuda`
- Curriculum: `32-48:200,48-64:400,64-80:800`
- Eval holes: `64,72,80,88,96`, `FULL_EVAL_N=256`
- Batch: `FULL_BATCH=48`

## Training Trace

| step | stage | ce | loop1 loss | loop last loss | unit loss |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | 32-48 | 0.2310 | 0.2250 | 0.2310 | 0.0193 |
| 200 | 32-48 | 0.1054 | 0.1361 | 0.1054 | 0.0105 |
| 300 | 48-64 | 0.1922 | 0.2465 | 0.1922 | 0.0215 |
| 400 | 48-64 | 0.1927 | 0.2544 | 0.1927 | 0.0229 |
| 500 | 48-64 | 0.1671 | 0.2310 | 0.1671 | 0.0187 |
| 600 | 48-64 | 0.1516 | 0.2259 | 0.1516 | 0.0174 |
| 700 | 64-80 | 0.2489 | 0.4311 | 0.2489 | 0.0272 |
| 800 | 64-80 | 0.2545 | 0.4227 | 0.2545 | 0.0271 |
| 900 | 64-80 | 0.2331 | 0.4564 | 0.2331 | 0.0227 |
| 1000 | 64-80 | 0.1671 | 0.4728 | 0.1671 | 0.0157 |
| 1100 | 64-80 | 0.1309 | 0.5480 | 0.1309 | 0.0118 |
| 1200 | 64-80 | 0.1103 | 0.5544 | 0.1103 | 0.0107 |
| 1300 | 64-80 | 0.0897 | 0.4814 | 0.0897 | 0.0084 |
| 1400 | 64-80 | 0.0873 | 0.4487 | 0.0873 | 0.0084 |

Training took `2253.1s`.

## Loop Depth

Eval at 80 holes:

| loop | exact | valid | solved | blank acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.7069 |
| 3 | 0.0586 | 0.0586 | 0.0586 | 0.9466 |
| 5 | 0.0625 | 0.0625 | 0.0625 | 0.9471 |
| 7 | 0.0625 | 0.0625 | 0.0625 | 0.9472 |

## Hole Transfer

Exact solve rate by loop7:

| holes | exact | valid | solved | blank acc |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.4297 | 0.4297 | 0.4297 | 0.9832 |
| 72 | 0.2031 | 0.2031 | 0.2031 | 0.9688 |
| 80 | 0.0625 | 0.0625 | 0.0625 | 0.9472 |
| 88 | 0.0039 | 0.0039 | 0.0039 | 0.9107 |
| 96 | 0.0000 | 0.0000 | 0.0000 | 0.8712 |

## Insight

h80 is now open. Extending the 64-80 stage changed h80 exact from `0.0` to `0.0625`, while h64 improved from `0.2305` to `0.4297` and h72 reached `0.2031`.

This answers the previous ambiguity: h80 was partly a curriculum frontier, not a hard architecture wall. The persistent unit-memory mechanism is strong enough to solve some 80-hole 16x16 boards when trained long enough on the hard stage.

The remaining frontier is around h88-h96. h88 exact is barely nonzero at `0.0039`, while h96 remains `0.0`. Blank accuracy is still high at h88/h96, so the remaining issue is global consistency under very sparse clues.

Loop depth remains necessary but saturates early. Loop1 exact is `0.0`; loop3 already reaches `0.0586`; loop5 and loop7 tie at `0.0625`.

## Decision

16x16 with persistent unit memory is supported through h80 under this budget.

Next high-ROI directions:

1. 25x25 conservative feasibility is now justified, but only at low/moderate holes and with strict kill criteria.
2. For 16x16, push h88/h96 only if using a stronger unit-token mechanism or a targeted hard curriculum; direct width/depth scaling is lower ROI.
3. True unit tokens inside the RWKV sequence remain the likely next architecture step if h88/h96 matter.

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
