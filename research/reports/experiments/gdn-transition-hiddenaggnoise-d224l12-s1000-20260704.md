# GDN Transition Hidden-Aggregate Noise Gate

## 1. Metainfo

- plan_id: P-DIAG-023
- run_name: gdn-transition-hiddenaggnoise-d224l12-s1000-20260704T1141Z-55e4237
- machine: AIStation GPU1 only
- status: discarded
- created_utc: 2026-07-04T11:41Z
- launched_utc: 2026-07-04T11:42Z
- aborted_utc: 2026-07-04T12:05:51Z

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
- launch SHA: `55e4237b5f483714b6115bc2448b8e59a4e618b7`
- worktree: `/huyang2/double-loop/.worktrees/pdiag023-hiddenagg-55e4237`

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
RUN_NAME=gdn-transition-hiddenaggnoise-d224l12-s1000-20260704T1141Z-55e4237 ./run.sh full
```

## 6. Artifacts

- remote run_dir: `/huyang2/double-loop/.worktrees/pdiag023-hiddenagg-55e4237/runs/gdn-transition-hiddenaggnoise-d224l12-s1000-20260704T1141Z-55e4237`
- local run_dir: `runs/gdn-transition-hiddenaggnoise-d224l12-s1000-20260704T1141Z-55e4237`
- logs: `runs/gdn-transition-hiddenaggnoise-d224l12-s1000-20260704T1141Z-55e4237/logs/run.log`
- abort: `runs/gdn-transition-hiddenaggnoise-d224l12-s1000-20260704T1141Z-55e4237/abort.json`
- checkpoint eval: `runs/gdn-transition-hiddenaggnoise-d224l12-s1000-20260704T1141Z-55e4237/output/checkpoint_eval_step000500.json`
- visualizations: none; stopped before final case-bank export because the checkpoint was already far below the clean curve.

## 7. Results

Stopped at step500.

| readout | loop1 exact | loop1 blank | loop5 exact | loop5 blank |
|---|---:|---:|---:|---:|
| holes53 | 0.0000 | 0.2572 | 0.0000 | 0.2571 |
| holes60 | 0.0000 | 0.2611 | 0.0000 | 0.2614 |
| holes64 | 0.0000 | 0.2617 | 0.0000 | 0.2594 |

Training diagnostics at step500:

- CE: `1.6824`
- loop1 CE: `1.6468`
- loop5 CE: `1.6824`
- `hidden_agg_noise_norm`: `61581.7344`
- `hidden_agg_noise_entropy`: `0.0000`
- `hidden_agg_noise_max_weight`: `1.0000`
- elapsed: `1395.7s`

## 8. Conclusions

Discarded.

The mechanism did not produce useful stochastic aggregation. It collapsed to selecting one token as the global summary on every update (`entropy=0`, `max_weight=1`), and the perturbation norm became enormous. This made the model behave close to random on hidden blanks: step500 blank accuracy stayed around `0.26`, far below the clean D224/L12 same-step transition curve.

The useful insight is narrow but solid: do not sweep `HIDDEN_AGG_NOISE_SCALE`. The failure is structural. If this idea is retried, the aggregation must be smoother and bounded, for example multi-token/top-k group aggregation with nonzero entropy, not one-hot Gumbel injection into every depth update.

## 9. Submission Record

No submission/tag planned unless the primary score is genuinely strong.
