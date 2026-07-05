# GDN Transition Soft-Group Aggregate Noise Gate

## 1. Metainfo

- plan_id: P-DIAG-024
- run_name: gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-pending
- machine: AIStation GPU1 only
- status: prelaunch
- created_utc: 2026-07-05T05:17Z

## 2. Hypothesis

P-DIAG-023 showed that one-hot Gumbel aggregation destroys optimization. That does not kill the broader noise idea. The more plausible version is bounded multi-token stochastic aggregation: sample a small group of hidden states, aggregate them softly, and cap the perturbation norm so noise regularizes global summary formation without turning into a single-cell shock.

Prediction: entropy should stay nonzero, perturbation should be clipped to a small norm, and step500 should stay above random blank accuracy. If it survives that gate and step1000 beats the clean D224/L12 same-step transition curve, this noise direction deserves scale. If not, stop aggregation noise for now.

## 3. Configuration

- backbone: GDN + native FutureSeed terminal-state seeding
- model: D224/L12/H14/head_dim16, `GDN_EXPAND_V=4.0`, `GDN_USE_SHORT_CONV=0`
- loops: `MAX_LOOPS=5`, `LOOP_LOSS=all`
- data: official EqR Sudoku extreme arrays
- curriculum: `46-50:100,51-55:900`
- batch: microbatch32, grad accumulation4, effective batch128
- new mechanism: `HIDDEN_AGG_NOISE_MODE=soft_group`, `HIDDEN_AGG_NOISE_SCALE=0.03`, `HIDDEN_AGG_NOISE_TEMP=4.0`, `HIDDEN_AGG_NOISE_TOPK=8`, `HIDDEN_AGG_NOISE_MAX_NORM=2.0`, `HIDDEN_AGG_NOISE_DETACH=1`
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
HIDDEN_AGG_NOISE_MODE=soft_group HIDDEN_AGG_NOISE_SCALE=0.03 HIDDEN_AGG_NOISE_TEMP=4.0 \
HIDDEN_AGG_NOISE_TOPK=8 HIDDEN_AGG_NOISE_MAX_NORM=2.0 HIDDEN_AGG_NOISE_DETACH=1 \
HOLE_STAGES=46-50:100,51-55:900 EVAL_HOLES=60 EVAL_HOLES_LIST=53,60,64 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=1000 FULL_EVAL_N=512 \
FULL_LOG_EVERY=100 FULL_ROLLOUT_KS=1 SAVE_TRAIN_CHECKPOINT_EVERY=100 \
EVAL_CHECKPOINT_STEPS=500,1000 EVAL_CHECKPOINT_HOLES_LIST=53,60,64 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
CASE_BANK_HOLES=53,60,64 CASE_BANK_N=4 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,3,5 \
RUN_NAME=gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-<sha> ./run.sh full
```

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

No tag/submission planned unless the primary score is genuinely strong.
