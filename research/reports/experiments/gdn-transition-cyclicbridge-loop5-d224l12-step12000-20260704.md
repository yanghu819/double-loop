# GDN Transition Cyclic Bridge Loop5 D224/L12 Step12000

## 1. Metainfo

- run_name: `gdn-transition-cyclicbridge-loop5-d224l12-step12000-20260704T0520Z-3f09ea5`
- plan_id: `P-DIAG-021`
- machine: AIStation `GPU1` only
- local_start_time: `2026-07-04 13:20 CST`
- status: `in-progress`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

P-DIAG-019 found the strongest current data-scaling signal: staged bridge-hard training pushed official `56-64` exact to `0.1250` while keeping `51-55` useful. P-DIAG-020 showed that simply adding loop8 and effective batch256 is too slow to be worth the GPU budget.

The hypothesis here is narrower and more practical: the same cyclic bridge-hard data distribution can keep moving the blank-count boundary if we use the proven loop5/effective batch128 training path and save frequent train checkpoints. This tests the bitter-lesson axis that has already produced slope: more examples, more wall-clock training, same generic backbone.

No Sudoku repair, no search, no selector, no oracle rollout, no task-specific rules.

## 3. Configuration

- Source SHA: `3f09ea5212146a7c64f4abbe9a905e6452d92697`.
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
RUN_NAME=gdn-transition-cyclicbridge-loop5-d224l12-step12000-20260704T0520Z-3f09ea5 ./run.sh full
```

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

No tag unless the primary score is meaningfully strong and mechanism conclusion is clean.
