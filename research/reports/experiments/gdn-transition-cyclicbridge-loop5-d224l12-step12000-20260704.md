# GDN Transition Cyclic Bridge Loop5 D224/L12 Step12000

## 1. Metainfo

- run_name: `gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5`
- plan_id: `P-DIAG-021`
- machine: AIStation `GPU1` only
- local_start_time: `2026-07-04 13:27 CST`
- status: `done`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

P-DIAG-019 found the strongest current data-scaling signal: staged bridge-hard training pushed official `56-64` exact to `0.1250` while keeping `51-55` useful. P-DIAG-020 showed that simply adding loop8 and effective batch256 is too slow to be worth the GPU budget.

The hypothesis here is narrower and more practical: the same cyclic bridge-hard data distribution can keep moving the blank-count boundary if we use the proven loop5/effective batch128 training path and save frequent train checkpoints. This tests the bitter-lesson axis that has already produced slope: more examples, more wall-clock training, same generic backbone.

No Sudoku repair, no search, no selector, no oracle rollout, no task-specific rules.

## 3. Configuration

- Source SHA: `9c621a56c8ff672c23934ccd2a5c1b2bb0c900f3`.
- Resume checkpoint: P-DIAG-019 best mixed-hard checkpoint `train_state_step008400.pt`.
- Data: official EqR Sudoku arrays.
- Backbone: native FutureSeed GDN, D224/L12/H14/D16, `GDN_EXPAND_V=4.0`.
- Loop: `MAX_LOOPS=5`, `LOOP_LOSS=all`.
- Effective batch: microbatch `32`, grad accumulation `4`, effective batch `128`.
- Active continuation after step8400:
  - `56-64:300`
  - `51-64:500`
  - `51-55:300`
  - `56-64:500`
  - `51-64:600`
  - `51-55:300`
  - `56-64:500`
  - `51-64:600`
- Target: `FULL_STEPS=12000`.
- Checkpoint eval: `9000,10000,11000,12000` on holes `53,60,64`.
- Train checkpoint safety: `SAVE_TRAIN_CHECKPOINT_EVERY=100`.
- Final official blank ranges: `46-50,51-55,56-64`.
- Case-bank loops: `1,3,5`.

## 4. Prediction And Kill Criteria

Prediction:

- If cyclic bridge-hard data is the right scale axis, official `56-64` loop5 exact should beat P-DIAG-019 final `0.1250`, ideally `>=0.18`, without pushing `51-55` below `0.25`.
- Holes60/64 checkpoint exact should show a non-flat trend at `9000,10000,11000,12000`.

Kill criteria:

- Stop on NaN/OOM.
- Stop if throughput is unexpectedly worse than P-DIAG-019 despite loop5/effective batch128.
- Discard if final official `51-55 <0.25` or `56-64 <=0.1250`.
- Discard if only blank accuracy improves while full-board exact stays flat.

## 5. Commands

Planned remote launch:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
SUDOKU_SIZE=9 PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
RESUME_TRAIN_CHECKPOINT=/huyang2/double-loop/.worktrees/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/runs/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/checkpoints/train_state_step008400.pt \
SAVE_TRAIN_CHECKPOINT_EVERY=100 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0 \
D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all SCRATCH_MODE=none \
HOLE_STAGES=46-50:100,51-55:5900,51-64:1000,51-55:600,51-64:800,56-64:300,51-64:500,51-55:300,56-64:500,51-64:600,51-55:300,56-64:500,51-64:600 \
EVAL_HOLES=60 EVAL_HOLES_LIST=53,60,64 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=12000 FULL_EVAL_N=512 \
FULL_ROLLOUT_KS="" FULL_LOG_EVERY=100 \
EVAL_CHECKPOINT_STEPS=9000,10000,11000,12000 EVAL_CHECKPOINT_HOLES_LIST=53,60,64 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
CASE_BANK_HOLES=53,60,64 CASE_BANK_N=4 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,3,5 \
RUN_NAME=gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5 ./run.sh full
```

## 6. Artifacts

- Remote run directory:
  `/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5`
- Local archived run directory:
  `runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5`
- Remote light archive:
  `/huyang2/double-loop/artifacts/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5-metadata-light.tgz`
- Local pulled archive:
  `.codex-transfer/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5-metadata-light.tgz`
- Key files:
  - `config.json`
  - `metadata.json`
  - `score.json`
  - `logs/run.log`
  - `output/checkpoint_eval_step009000.json`
  - `output/checkpoint_eval_step010000.json`
  - `output/checkpoint_eval_step011000.json`
  - `output/checkpoint_eval_step012000.json`
  - `output/futureseed_loop_seed52.json`
  - `output/futureseed_loop_seed52.md`
  - `output/futureseed_loop_case_seed52.html`
  - `output/case_bank/official_b46_50/index.html`
  - `output/case_bank/official_b51_55/index.html`
  - `output/case_bank/official_b56_64/index.html`
  - `visualizations/index.html`

## 7. Results

Final score:

| Metric | Value |
|---|---:|
| `metrics.eval_clean.loop5.label_exact` | `0.236328` |
| loop1 exact / blank | `0.023438 / 0.538128` |
| loop2 exact / blank | `0.054688 / 0.591203` |
| loop3 exact / blank | `0.189453 / 0.617985` |
| loop4 exact / blank | `0.226562 / 0.625607` |
| loop5 exact / blank | `0.236328 / 0.628719` |
| loop5 - loop1 exact gain | `+0.212891` |
| `fs_gate_mean` | `0.437187` |
| `fs_state_norm` | `13.9896` |

Official blank-range readout:

| Blank range | loop5 exact | loop5 blank_acc |
|---|---:|---:|
| `46-50` | `1.0000` | `1.0000` |
| `51-55` | `0.3555` | `0.7312` |
| `56-64` | `0.1582` | `0.5647` |

Checkpoint eval trajectory:

| Step | holes53 loop5 exact / blank | holes60 loop5 exact / blank | holes64 loop5 exact / blank |
|---:|---:|---:|---:|
| 9000 | `0.2070 / 0.6449` | `0.2480 / 0.6617` | `0.2129 / 0.6329` |
| 10000 | `0.2188 / 0.6307` | `0.2188 / 0.6422` | `0.2188 / 0.6220` |
| 11000 | `0.2012 / 0.6212` | `0.2148 / 0.6373` | `0.2441 / 0.6295` |
| 12000 | `0.1992 / 0.6098` | `0.2441 / 0.6432` | `0.2324 / 0.6134` |

Case-bank summary:

| Bank | exact | blank_acc | visual buckets |
|---|---:|---:|---|
| `official_b46_50` | `1.0000` | `1.0000` | 4 solved-by-loop |
| `official_b51_55` | `0.3711` | `0.7287` | 4 solved-by-loop, 4 almost-solved, 4 hard-failure |
| `official_b56_64` | `0.1211` | `0.5644` | 4 solved-by-loop, 4 almost-solved, 4 hard-failure |

## 8. Conclusions

Decision: mark P-DIAG-021 as `done`.

This run passes the planned success gate. Official `51-55` exact is `0.3555`, well above the `0.25` floor, and official `56-64` exact is `0.1582`, above P-DIAG-019's `0.1250`. The cleanest positive signal is loop computation itself: final exact jumps from `0.0234` at loop1 to `0.2363` at loop5.

The harder lesson is that this is not a simple "train longer and it monotonically rises" story. Holes60 is already high at step9000 (`0.2480`) and returns near that at step12000 (`0.2441`); holes64 peaks at step11000 (`0.2441`) then eases at step12000 (`0.2324`). Stage CE can become tiny inside a hard segment while full-board exact still oscillates. The model has enough local token skill to solve many boards, but stable global consistency over the hardest blank range is still the bottleneck.

Next decision: do not run the same-shape cyclic continuation again as a blind 15000/18000-step repeat. Higher ROI is one of:

- more effective recurrent state/capacity per FLOP;
- a stability-oriented bridge-hard curriculum with explicit best-checkpoint selection;
- a simple generic state/update mechanism that improves board-level consistency without Sudoku repair/search/selector.

## 9. Submission Record

No tag unless the primary score is meaningfully strong and mechanism conclusion is clean.
