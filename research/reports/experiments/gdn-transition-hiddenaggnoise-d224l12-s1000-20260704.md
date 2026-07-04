# GDN Transition Hidden-Aggregate Noise Gate

## 1. Metainfo

- plan_id: P-DIAG-023
- run_name: gdn-transition-hiddenaggnoise-d224l12-s1000-20260704T1141Z-pending
- machine: AIStation GPU1 only
- status: prelaunch
- created_utc: 2026-07-04T11:41Z

## 2. Hypothesis

Direct feature noise was the wrong object: it perturbs hidden coordinates, not the way recurrent state builds a global summary. A Gumbel-style stochastic hidden aggregation during training may force FutureSeed+loop to learn a more robust global attractor and open the 51-55 blank transition earlier.

Prediction: if the idea is real, by step1000 holes53/60 loop5 exact or blank accuracy should beat the clean D224/L12 same-step curve, and official 51-55 should show nonzero exact. If it only raises training noise without exact/blank gain, stop this direction instead of sweeping noise scales.

## 3. Configuration

- backbone: GDN + native FutureSeed terminal-state seeding
- model: D224/L12/H14/head_dim16, `GDN_EXPAND_V=4.0`, `GDN_USE_SHORT_CONV=0`
- loops: `MAX_LOOPS=5`, `LOOP_LOSS=all`
- data: official EqR Sudoku extreme arrays
- curriculum: `46-50:100,51-55:900`
- batch: microbatch32, grad accumulation4, effective batch128
- new mechanism: `HIDDEN_AGG_NOISE_SCALE=0.03`, `HIDDEN_AGG_NOISE_TEMP=1.0`, `HIDDEN_AGG_NOISE_DETACH=1`
- disabled: feature-diff noise, scratch, repair, search, selector, seed sweep

## 4. Environment

- remote work_dir: `/huyang2/double-loop`
- GPU: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Python: `/opt/conda/bin/python`
- repo branch: `codex/gpu1-experiment-tracking`
- launch SHA: pending post-commit

## 5. Commands

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
PYTHON_BIN=/opt/conda/bin/python SUDOKU_SIZE=9 \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0 \
D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all SCRATCH_MODE=none \
HIDDEN_AGG_NOISE_SCALE=0.03 HIDDEN_AGG_NOISE_TEMP=1.0 HIDDEN_AGG_NOISE_DETACH=1 \
HOLE_STAGES=46-50:100,51-55:900 EVAL_HOLES=60 EVAL_HOLES_LIST=53,60,64 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=1000 FULL_EVAL_N=512 \
FULL_LOG_EVERY=100 SAVE_TRAIN_CHECKPOINT_EVERY=100 \
EVAL_CHECKPOINT_STEPS=500,1000 EVAL_CHECKPOINT_HOLES_LIST=53,60,64 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
CASE_BANK_HOLES=53,60,64 CASE_BANK_N=4 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,3,5 \
RUN_NAME=gdn-transition-hiddenaggnoise-d224l12-s1000-20260704T1141Z-<sha> ./run.sh full
```

## 6. Artifacts

- remote run_dir: pending
- local run_dir: pending
- logs: pending
- visualizations: pending

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

No submission/tag planned unless the primary score is genuinely strong.
