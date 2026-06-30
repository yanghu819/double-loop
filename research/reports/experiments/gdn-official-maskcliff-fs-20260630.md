# GDN FutureSeed Official Sudoku Mask-Cliff Diagnostic

## 1. Metainfo

- Plan ID: `P-DIAG-003`
- Status: done
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-06-30 21:57:06 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: `74ed7591d9dba41ccdea0b6c4638112e3bbc4ad5`
- Run name: `gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759`

## 2. Hypothesis

P-DIAG-002 showed that full RWKV+native FutureSeed almost solves the lower
official blank bucket but hits a hard exact cliff above about 50 blanks. This
run asks whether GDN+FutureSeed has the same cliff shape.

Prediction:

- If GDN+FutureSeed also solves `46-50` blanks but fails `51+`, the bottleneck
  is the current FutureSeed+loop global-consistency mechanism, not just GDN.
- If GDN+FutureSeed fails even `46-50`, then GDN state/backbone is an additional
  bottleneck relative to RWKV.

This is one diagnostic run, not an ablation table. It should not change loss,
seed, width, or dynamics.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state FutureSeed
- Architecture: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- Train: `FULL_STEPS=1000`, `FULL_BATCH=128`, `BLANK_LOSS_WEIGHT=8`
- Eval: official test split, `FULL_EVAL_N=512`
- Official blank-range eval: `46-50,51-55,56-64`
- Case-bank visualization: per blank-range groups, loops `1,2,3,5`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule or postprocessing, no no-FS arm in this run.

## 4. Environment

- Local branch: `codex/gpu1-experiment-tracking`
- Remote row: AIStation `GPU1` only; `GPU2` was not started.
- GPU: `NVIDIA A100-SXM4-80GB`
- Python/PyTorch: `/opt/conda/bin/python`, `torch 2.7.0+cu126`
- Remote detached worktree:
  `/huyang2/double-loop/.worktrees/gdn-maskcliff-74ed759-20260630T1357Z`
- Remote run dir:
  `/huyang2/double-loop/.worktrees/gdn-maskcliff-74ed759-20260630T1357Z/runs/gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759`

## 5. Commands

Local static validation only:

```bash
python -m py_compile experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py
bash -n run.sh
```

Remote launch shape, shortened from 1500 to 1000 steps because the active GPU1
lease had about 32 minutes remaining at launch:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
PIP_CACHE_DIR=/huyang2/double-loop/.cache/pip \
HF_HOME=/huyang2/double-loop/.cache/huggingface \
TORCH_HOME=/huyang2/double-loop/.cache/torch \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
D_MODEL=192 LAYERS=10 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 FULL_STEPS=1000 FULL_BATCH=128 FULL_EVAL_N=512 \
HOLES_MIN=46 HOLES_MAX=64 EVAL_HOLES=56 EVAL_HOLES_LIST=56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=500,1000 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=1 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- If step100 takes more than 15 minutes, stop by exact PID and relaunch once at
  `FULL_BATCH=96`.
- If step500 blank accuracy remains near the no-FS plateau (`~0.27`) and CE is
  still high, stop and classify GDN as not open under this budget.
- If run completes, judge only by blank-range shape, not mixed exact alone.

Actual launch:

```bash
bash /huyang2/double-loop/artifacts/launch/gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759/launch.sh
```

## 6. Artifacts

- Local browsable run:
  `runs/gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759`
- Git-tracked archive:
  `research/reports/visualizations/gdn-official-maskcliff-fs-20260630`
- Visual dashboard:
  `runs/gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759/visualizations/index.html`
- Case-bank HTML:
  - `output/case_bank/official_b46_50/index.html`
  - `output/case_bank/official_b51_55/index.html`
  - `output/case_bank/official_b56_64/index.html`
- Metrics:
  - `output/futureseed_loop_seed52.json`
  - `output/checkpoint_eval_step000500.json`
  - `output/checkpoint_eval_step001000.json`
- Logs: `logs/run.log`, `nohup.out`, `launch.env`
- Source: `source_HEAD.txt`, `source.patch`, `source_snapshot.tar.gz`
- Checkpoints were intentionally not pulled or committed.

## 7. Results

Training curve:

| step | CE | loop1 loss | loop-last loss |
|---:|---:|---:|---:|
| 100 | 1.8153 | 1.8158 | 1.8153 |
| 200 | 1.2091 | 1.2939 | 1.2091 |
| 300 | 1.1216 | 1.2952 | 1.1216 |
| 400 | 1.0812 | 1.3305 | 1.0812 |
| 500 | 1.0398 | 1.2423 | 1.0398 |
| 600 | 1.0376 | 1.2668 | 1.0376 |
| 700 | 1.0455 | 1.2545 | 1.0455 |
| 800 | 1.0234 | 1.2437 | 1.0234 |
| 900 | 1.0221 | 1.2030 | 1.0221 |
| 1000 | 1.0091 | 1.1829 | 1.0091 |

Checkpoint eval on `holes56`:

| checkpoint | elapsed sec | loop5 exact | loop5 blank acc |
|---|---:|---:|---:|
| step500 | 322.2 | 0.0156 | 0.5001 |
| step1000 | 615.6 | 0.0156 | 0.5184 |

Final mixed official eval:

| loop | exact | blank acc |
|---:|---:|---:|
| 1 | 0.0000 | 0.4667 |
| 2 | 0.0234 | 0.5019 |
| 3 | 0.0234 | 0.5233 |
| 4 | 0.0234 | 0.5252 |
| 5 | 0.0234 | 0.5252 |

Final official blank-range eval:

| blank range | loop1 exact | loop5 exact | loop1 blank acc | loop5 blank acc | loop gain |
|---|---:|---:|---:|---:|---:|
| 46-50 | 0.0742 | 0.9863 | 0.9359 | 0.9993 | +0.9121 |
| 51-55 | 0.0000 | 0.0000 | 0.4817 | 0.5457 | +0.0000 |
| 56-64 | 0.0000 | 0.0000 | 0.4373 | 0.4879 | +0.0000 |

Case-bank summary:

| blank range | exact | blank acc | selected cases |
|---|---:|---:|---|
| 46-50 | 0.9805 | 0.9985 | solved-by-loop 1, almost-solved 1 |
| 51-55 | 0.0000 | 0.5336 | hard-failure 1 |
| 56-64 | 0.0000 | 0.4886 | hard-failure 1 |

Peak observed memory during training was about `49.6GB`; utilization reached
about `90%`. Step100 completed well below the 15 minute kill threshold.

## 8. Conclusions

This run answers the user's GDN question clearly:

- GDN+native FutureSeed is not simply broken. On `46-50` blanks, loop5 reaches
  `0.9863` exact and `0.9993` blank accuracy.
- GDN+native FutureSeed shows the same qualitative cliff as RWKV+native
  FutureSeed: exact collapses to `0` at `51-55` and `56-64` blanks.
- Loop helps strongly inside the easy bucket (`46-50` exact `0.0742 -> 0.9863`)
  but gives no exact gain past the cliff (`51+` stays `0` exact). It improves
  soft blank accuracy without restoring board-level consistency.
- The right interpretation is not "GDN cannot use FutureSeed." It can. The
  hard bottleneck is the current FutureSeed+loop state/global-consistency
  mechanism when the number of missing cells crosses roughly 50.

Decision:

- Mark this diagnostic done.
- Do not run a no-FS/GDN width/seed/loss table for this question.
- Next high-ROI work should target the `51-55` transition with a generic,
  scalable state formulation or training pressure that converts per-cell
  confidence into full-board consistency. Keep avoiding selector/search/repair
  and Sudoku-specific rules.

## 9. Submission Record

Not a submission run.
