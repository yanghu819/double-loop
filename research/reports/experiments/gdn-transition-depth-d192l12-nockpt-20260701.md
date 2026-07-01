# GDN FutureSeed D192/L12 No-Checkpoint Isolation

## 1. Metainfo

- Plan ID: `P-DIAG-008`
- Status: stopped / superseded
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-07-01 16:05:00 +0800`
- Stopped time: `2026-07-01 16:30:00 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: `46cfd669ff1ff2821d16b7e13cc5bd6dd1ba02c2`
- Run name: `gdn-transition-depth-d192l12-nockpt-expv4-s3000-20260701T0805Z-46cfd66`
- Parent evidence:
  P-DIAG-006 and P-DIAG-007 NaN boundaries

## 2. Hypothesis

P-DIAG-006 and P-DIAG-007 both reached NaN at step100, and both used
`ACTIVATION_CHECKPOINT=1`. The stable D192/L10 expand_v4 transition runs did
not use activation checkpointing.

Mechanism question: are the NaNs caused by activation checkpointing around the
custom GDN recurrent path, or by the 12-layer expand_v4 depth/optimizer setup
itself?

Prediction:

- If activation checkpointing is the culprit, D192/L12 no-checkpoint with a
  smaller microbatch should have finite step100 CE.
- If depth/optimizer is the culprit, the no-checkpoint run will still NaN or
  OOM. Then depth scaling should stop until the optimizer/state formulation is
  made stable.

This is a one-bit infrastructure/scaling diagnosis, not a hyperparameter table.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=192`, `LAYERS=12`, `HEADS=12`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- Curriculum: `HOLE_STAGES=46-50:500,51-55:2500`
- Train: `FULL_STEPS=3000`, microbatch `FULL_BATCH=48`,
  `GRAD_ACCUM_STEPS=3`, effective batch `144`
- Eval: official test split, `FULL_EVAL_N=1024`
- Checkpoint eval: steps `1000,2000,3000`, holes53
- Official blank-range eval: `46-50,51-55,56-64`
- Case-bank visualization: loops `1,2,3,5`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`, no activation checkpoint
- Optimization: `LR=0.0015`, `WEIGHT_DECAY=0.001`, `BLANK_LOSS_WEIGHT=8`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule, no LR/seed/width/depth table.

## 4. Environment

- AIStation row: `GPU1`
- GPU: NVIDIA A800-SXM4-80GB
- Python: `/opt/conda/bin/python`
- Torch: `2.7.0+cu126`
- CUDA visible devices: `0`
- Remote worktree:
  `/huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-nockpt-46cfd66-20260701T0805Z`

## 5. Commands

Remote launch shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
PIP_CACHE_DIR=/huyang2/double-loop/.cache/pip \
HF_HOME=/huyang2/double-loop/.cache/huggingface \
TORCH_HOME=/huyang2/double-loop/.cache/torch \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=192 LAYERS=12 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 HOLE_STAGES=46-50:500,51-55:2500 \
FULL_STEPS=3000 FULL_BATCH=48 GRAD_ACCUM_STEPS=3 FULL_EVAL_N=1024 \
HOLES_MIN=46 HOLES_MAX=55 EVAL_HOLES=53 EVAL_HOLES_LIST=50,53,56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=1000,2000,3000 EVAL_CHECKPOINT_HOLES_LIST=53 \
SAVE_TRAIN_CHECKPOINT_EVERY=1000 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 LR=0.0015 WEIGHT_DECAY=0.001 \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=2 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- If CUDA OOMs, stop exact PID and archive; do not reduce batch repeatedly.
- If step100 is NaN, stop exact PID and archive; depth/optimizer is unstable.
- If step100 takes more than `20` minutes, stop as throughput-low-ROI.
- If step1000 holes53 loop5 exact is below D192/L10 expand_v4 step1000
  (`0.0205`) and blank_acc/CE are not better, stop unless near checkpoint.

Success criteria:

- Minimal success: finite step100 CE. This isolates checkpointing as likely
  culprit for P-DIAG-006/007.
- Strong success: holes53 loop5 exact `>= 0.06` by step3000 or official
  `51-55` loop5 exact `>= 0.05`.

## 6. Artifacts

- Local metadata:
  `runs/gdn-transition-depth-d192l12-nockpt-expv4-s3000-20260701T0805Z-46cfd66/`
- Remote run dir:
  `/huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-nockpt-46cfd66-20260701T0805Z/runs/gdn-transition-depth-d192l12-nockpt-expv4-s3000-20260701T0805Z-46cfd66`
- Archived files: `config.json`, `logs/run.log`, `source_HEAD.txt`,
  `source.patch`, `source_snapshot.tar.gz`, `abort.json`
- No final `score.json`: the run was intentionally stopped before eval because
  the training objective was superseded.

## 7. Results

This run answered the stability question but should not be used as the quality
scale result.

Training log:

| Step | Stage | Loop-last CE | Loop1 CE | Note |
|---:|---|---:|---:|---|
| 100 | 46-50 | 1.3685 | 1.3722 | finite |
| 200 | 46-50 | 0.1172 | 0.5490 | finite |
| 300 | 46-50 | 0.0339 | 0.3720 | finite |
| 400 | 46-50 | 0.0102 | 0.3998 | finite |
| 500 | 46-50 | 0.0208 | 0.3084 | finite |
| 600 | 51-55 | 1.0316 | 1.4687 | finite after hard-stage switch |
| 700 | 51-55 | 0.9808 | 1.4265 | finite; kill had a short delay |

Key observation:

- D192/L12 `GDN_EXPAND_V=4.0` without activation checkpointing stayed finite
  through step700.
- P-DIAG-006 and P-DIAG-007 both reached NaN at step100 with
  `ACTIVATION_CHECKPOINT=1`.
- Therefore the prior NaN failures are more likely an activation-checkpoint /
  custom GDN recurrent-backward interaction than a proof that L12 depth is
  impossible.

## 8. Conclusions

Stopped after user identified a more important training-objective mismatch:
EqR supervises every recurrent step, while this standalone run used
`LOOP_LOSS=final`.

Decision:

- Keep this run only as stability evidence.
- Do not compare its quality against EqR or use it as the main scale result.
- Relaunch the same no-checkpoint D192/L12/expand_v4 scale probe with
  `LOOP_LOSS=all`, so the training recipe matches EqR's per-loop supervision.

Lesson:

Every-loop supervision is not a Sudoku-specific trick. It is a generic
recurrent-training pressure and should be the default fair recipe when comparing
against EqR.

## 9. Submission Record

Not a submission run.
