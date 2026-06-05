# 16x16 Persistent Unit-Memory Probe

Purpose: test whether row/column/box state needs to persist across outer loops, rather than being recomputed as a pooled one-shot broadcast each loop.

Source SHA on GPU1: `79004cf56c965ae6e79070bd93bbf350f0da10ca`

Run: `frontier-16x16-unitmem-d192-l10-loop7-h80-b48-20260605T0803Z-79004cf`

Recorded UTC: `2026-06-05T08:35:25.523045+00:00`

## Setup

- GPU row: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Board: 16x16 Sudoku, 4x4 boxes, random holes
- Model: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=7`, final-loop CE
- Unit state: `UNIT_STATE_SCALE=1.0`, `UNIT_STATE_GATE_BIAS=-1.0`
- Unit memory: `UNIT_STATE_MODE=memory`, `UNIT_STATE_MEMORY_DECAY=0.7`, `UNIT_STATE_TOKEN_SCALE=1.0`
- Unit loss: disabled, `UNIT_LOSS_WEIGHT=0`
- FutureSeed: `FUTURE_SEED_SCALE=1`
- Active rollout noise: disabled, `NOISE_SCALE=0`, `ROLLOUT_NOISE_SCALE=0`
- Kernel: `RWKV_KERNEL=cuda`
- Curriculum: `32-48:300,48-64:500,64-80:400`
- Eval holes: `48,64,80,96`, `FULL_EVAL_N=256`
- Batch: `FULL_BATCH=48`

## Training Trace

| step | stage | ce | loop1 loss | loop last loss | unit loss |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | 32-48 | 0.2310 | 0.2250 | 0.2310 | 0.0193 |
| 200 | 32-48 | 0.1054 | 0.1361 | 0.1054 | 0.0105 |
| 300 | 32-48 | 0.1088 | 0.1289 | 0.1088 | 0.0104 |
| 400 | 48-64 | 0.1903 | 0.2395 | 0.1903 | 0.0231 |
| 500 | 48-64 | 0.1751 | 0.2337 | 0.1751 | 0.0204 |
| 600 | 48-64 | 0.1575 | 0.2588 | 0.1575 | 0.0181 |
| 700 | 48-64 | 0.1455 | 0.2622 | 0.1455 | 0.0162 |
| 800 | 48-64 | 0.1352 | 0.2737 | 0.1352 | 0.0151 |
| 900 | 64-80 | 0.2293 | 0.4378 | 0.2293 | 0.0243 |
| 1000 | 64-80 | 0.2081 | 0.4503 | 0.2081 | 0.0216 |
| 1100 | 64-80 | 0.1742 | 0.4814 | 0.1742 | 0.0157 |
| 1200 | 64-80 | 0.1226 | 0.5101 | 0.1226 | 0.0120 |

Training took `1855.8s`.

## Loop Depth

Eval at 64 holes:

| loop | exact | valid | solved | blank acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.8027 |
| 3 | 0.2344 | 0.2344 | 0.2344 | 0.9683 |
| 5 | 0.2305 | 0.2305 | 0.2305 | 0.9684 |
| 7 | 0.2305 | 0.2305 | 0.2305 | 0.9684 |

## Hole Transfer

Exact solve rate by loop7:

| holes | exact | valid | solved | blank acc |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 0.6992 | 0.7070 | 0.7070 | 0.9896 |
| 64 | 0.2305 | 0.2305 | 0.2305 | 0.9684 |
| 80 | 0.0000 | 0.0000 | 0.0000 | 0.9174 |
| 96 | 0.0000 | 0.0000 | 0.0000 | 0.8343 |

## Insight

Persistent unit memory is a real structural improvement. It uses a smaller model than the D256/L12 pooled unit-state run but substantially improves the harder-hole frontier:

- h48 exact: `0.4180` pooled D256 -> `0.6992` unit memory D192
- h64 exact: `0.0430` pooled D256 -> `0.2305` unit memory D192
- final train CE: `0.1867` pooled D256 -> `0.1226` unit memory D192
- train time: `2663.1s` pooled D256 -> `1855.8s` unit memory D192

This means the bottleneck was not just width, depth, or batch. The row/column/box state needs its own recurrent memory across loops.

Loop compute still matters: loop1 exact is `0.0`, while loop3 already reaches `0.2344`. Extra loops saturate after loop3-5 at eval, so the next improvement should strengthen unit-token reasoning rather than simply increasing `MAX_LOOPS`.

h80 remains closed: exact `0.0` despite blank accuracy `0.9174`. That is now the frontier. The model recognizes many blanks locally, but full-board coupling still fails when the missing set reaches 80 cells.

## Decision

Promote `UNIT_STATE_MODE=memory` to the larger-Sudoku mainline.

Do not jump to 25x25 yet. The next high-ROI experiment is a 16x16 h80-focused curriculum with memory mode:

- keep D192/L10/loop7 memory mode;
- extend the 64-80 stage instead of widening;
- evaluate h64/h72/h80/h88 to see whether h80 is a curriculum-length boundary or a remaining architecture boundary.

If h80 stays zero after a longer memory curriculum, implement true unit tokens inside the RWKV sequence instead of pooled EMA memory.

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
