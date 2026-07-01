# GDN FutureSeed D192/L12 All-Loop Scaling

## 1. Metainfo

- Plan ID: `P-DIAG-009`
- Status: discarded / stopped after checkpoint1000
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-07-01 16:36:00 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: `13921d16166c1d7464d26e80de26dd2dde7c0a53`
- Run name: `gdn-transition-depth-d192l12-allloop-expv4-s3000-20260701T0845Z-13921d1`
- Parent evidence:
  P-DIAG-008 finite-through-step700 no-checkpoint stability result; user
  correction that EqR uses per-loop supervision.

## 2. Hypothesis

EqR supervises every recurrent step. Our standalone FutureSeed/GDN scaling runs
have mostly used final-loop supervision, which gives sparser gradients and may
let early recurrent states become poor answer states.

Mechanism question:

Can EqR-aligned every-loop supervision improve the `51-55` blank transition by
making each loop produce a readable, board-consistent state, instead of asking
only the final loop to fix the board?

Prediction:

- If sparse final-loop supervision was a major bottleneck, `LOOP_LOSS=all`
  should improve holes53 and official `51-55` exact earlier than P-DIAG-004/005,
  while preserving loop1-to-loop5 refinement.
- If all-loop supervision only makes every loop copy the same operating point,
  exact will remain near the prior plateau and loop gain will shrink. Then the
  bottleneck is state/update capacity, not loss placement.

This is not a loss-weight table. It is one recipe-alignment probe against EqR.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=192`, `LAYERS=12`, `HEADS=12`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`
- Loop supervision: `LOOP_LOSS=all`
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
  no Sudoku-specific rule, no seed/LR/loss-weight sweep.

## 4. Environment

- AIStation row: `GPU1`
- GPU: NVIDIA A800-SXM4-80GB
- Python: `/opt/conda/bin/python`
- Torch: `2.7.0+cu126`
- CUDA visible devices: `0`
- Remote worktree:
  `/huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-allloop-13921d1-20260701T0845Z`
- Launch dir:
  `/huyang2/double-loop/artifacts/launch/gdn-transition-depth-d192l12-allloop-expv4-s3000-20260701T0845Z-13921d1`
- Startup validation: process command contains `--loop_loss all`; GPU memory
  about `57.8GB`; no GPU2 process.

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
L_CYCLES=2 MAX_LOOPS=5 LOOP_LOSS=all \
HOLE_STAGES=46-50:500,51-55:2500 \
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

- If CUDA OOMs, stop exact PID and archive.
- If step100 is NaN, stop exact PID and archive.
- If step1000 holes53 loop5 exact is below D192/L10 expand_v4 step1000
  (`0.0205`) and blank_acc/CE are not better, stop unless near checkpoint.
- If loop1 and loop5 become identical while exact remains low, classify as
  all-loop-copy failure rather than continue same recipe longer.

Success criteria:

- Strong: official `51-55` loop5 exact `>= 0.05`, or holes53 loop5 exact
  `>= 0.06` by step3000.
- Mechanism-positive: all-loop improves holes53 step1000/2000 vs D192/L10
  expand_v4 and preserves loop gain without relying on selector/repair.

## 6. Artifacts

- Remote run dir:
  `/huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-allloop-13921d1-20260701T0845Z/runs/gdn-transition-depth-d192l12-allloop-expv4-s3000-20260701T0845Z-13921d1`
- Eval-only visualization run:
  `runs/gdn-transition-depth-d192l12-allloop-step1000-eval-20260701T0930Z-13921d1/`
- Local launch script:
  `artifacts/launch/gdn-transition-depth-d192l12-allloop-expv4-s3000-20260701T0845Z-13921d1/launch.sh`
- Training checkpoint eval:
  `runs/gdn-transition-depth-d192l12-allloop-expv4-s3000-20260701T0845Z-13921d1/output/checkpoint_eval_step001000.json`
- Eval-only score:
  `runs/gdn-transition-depth-d192l12-allloop-step1000-eval-20260701T0930Z-13921d1/score.json`
- Eval visualization:
  `runs/gdn-transition-depth-d192l12-allloop-step1000-eval-20260701T0930Z-13921d1/visualizations/index.html`
- Case banks:
  `output/case_bank/official_b46_50/`,
  `output/case_bank/official_b51_55/`,
  `output/case_bank/official_b56_64/` under the eval-only run.

## 7. Results

Training was stopped at checkpoint1000 because the result already answered the
mechanism question and continuing to step2000 was low ROI.

Training curve:

| Step | Stage | Loop-last CE | Total CE | Loop1 CE | Note |
|---:|---|---:|---:|---:|---|
| 100 | 46-50 | 1.4117 | 1.4108 | 1.4094 | finite, all-loop active |
| 200 | 46-50 | 0.0979 | 0.1063 | 0.1340 | loop1 trained |
| 300 | 46-50 | 0.0183 | 0.0215 | 0.0332 | loop1 much better than final-only run |
| 400 | 46-50 | 0.0149 | 0.0167 | 0.0242 | stable |
| 500 | 46-50 | 0.0020 | 0.0045 | 0.0134 | easy bucket solved |
| 600 | 51-55 | 1.0038 | 1.0071 | 1.0194 | hard-stage switch finite |
| 700 | 51-55 | 0.9617 | 0.9660 | 0.9815 | hard-stage CE improves |
| 800 | 51-55 | 0.9529 | 0.9591 | 0.9810 | plateau-ish |
| 900 | 51-55 | 0.9628 | 0.9677 | 0.9856 | no clear slope |
| 1000 | 51-55 | 0.9291 | 0.9362 | 0.9602 | checkpoint eval |

Checkpoint1000 holes53:

| Loop | Exact | Blank Acc |
|---:|---:|---:|
| 1 | 0.0205 | 0.5067 |
| 2 | 0.0215 | 0.5203 |
| 3 | 0.0215 | 0.5215 |
| 4 | 0.0215 | 0.5218 |
| 5 | 0.0215 | 0.5216 |

Eval-only run from the step1000 checkpoint:

| Loop | Exact | Blank Acc |
|---:|---:|---:|
| 1 | 0.0283 | 0.5100 |
| 2 | 0.0283 | 0.5243 |
| 3 | 0.0283 | 0.5244 |
| 4 | 0.0283 | 0.5249 |
| 5 | 0.0283 | 0.5245 |

Official blank-range eval at step1000:

| Range | Exact | Blank Acc | Case-bank selection |
|---|---:|---:|---|
| 46-50 | 0.9971 | 0.9999 | 3 solved-by-loop, 2 almost solved |
| 51-55 | 0.0000 | 0.5447 | 3 hard failures |
| 56-64 | 0.0000 | 0.4927 | 3 hard failures |

Comparison to final-loop P-DIAG-008:

- All-loop massively improves intermediate loop CE. At step300, final-loop
  loop1 CE was `0.3720`; all-loop loop1 CE is `0.0332`.
- At hard-stage step600, final-loop loop1/last CE was `1.4687/1.0316`;
  all-loop is `1.0194/1.0038`.
- This improvement does not convert into late-loop exact improvement. Eval-only
  exact is identical from loop1 through loop5 (`0.0283`), and checkpoint holes53
  loop5 is only `0.0215`.

## 8. Conclusions

Decision: discard as a scaling direction, keep as a mechanism lesson.

What worked:

- EqR-style every-loop supervision is active and useful as a training recipe.
- It makes every loop's state readable and dramatically improves early-loop CE.
- It remains stable on the D192/L12 no-checkpoint expand_v4 configuration.

What did not work:

- It does not solve the 51-55 blank transition.
- It mostly moves the same operating point earlier: loop1 is already near
  loop5, and loop5 adds almost no full-board exact.
- Continuing this same recipe to step2000/3000 would likely be low ROI compared
  with changing the state/update capacity.

Lesson:

Every-loop CE should become the fair default when comparing to EqR, but it is
not the missing board-level consistency mechanism. The next high-ROI work is not
another loop-loss variant. It should target a more scalable recurrent state or
update rule that keeps the benefit of readable intermediate states while giving
later loops a real way to change global decisions.

## 9. Submission Record

Not a submission run.
