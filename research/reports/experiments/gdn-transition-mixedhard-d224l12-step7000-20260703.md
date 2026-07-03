# GDN Transition Mixed Hard D224/L12 Step7000

## 1. Metainfo

- run_name: `gdn-transition-mixedhard-d224l12-step7000-20260703`
- plan_id: `P-DIAG-018`
- machine: AIStation `GPU1` only
- local_start_time: `2026-07-03 10:13 CST`
- status: `in-progress`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

P-DIAG-017 showed the cleanest current positive signal: a short mixed `51-64` continuation from the D224/L12 step6000 checkpoint improved holes60 more than the much longer pure `56-64` tail, while preserving holes53.

The mechanism hypothesis is simple and bitter-lesson-compatible: after the model has opened `51-55`, a broader hard-data distribution gives the recurrent state enough continuous support to learn harder global consistency. If this is true, longer mixed `51-64` training from step6100 should improve holes60 / official `56-64` without erasing `51-55`. If it fails, same-shape long training is no longer the main path.

## 3. Configuration

- Resume checkpoint: P-DIAG-017 step6100 checkpoint.
- Data: official EqR Sudoku arrays.
- Backbone: native FutureSeed GDN, D224/L12/H14/D16, `GDN_EXPAND_V=4.0`.
- Loop: loop5, `LOOP_LOSS=all`.
- Continuation distribution: `HOLES_MIN=51`, `HOLES_MAX=64`.
- Target: `FULL_STEPS=7000`, with checkpoint eval at `6300,6600,7000`.
- No scratch, no extra loss, no repair/search/selector, no CPU smoke, no GPU2.

Kill criteria:

- At step6300, stop if holes60 loop5 is clearly below step6100 (`0.1973/0.6603`) and holes53 also drops, because that would mean the short positive signal was not stable.
- Stop immediately on NaN/OOM/low GPU utilization with high memory.

## 4. Environment

- remote work_dir: `/huyang2/double-loop/.worktrees/<run-worktree>`
- source SHA: to be filled from the launch commit.
- GPU: AIStation `GPU1`, `CUDA_VISIBLE_DEVICES=0`.
- Python: `/opt/conda/bin/python`.
- Repo path: `/huyang2/double-loop`.

## 5. Commands

Planned launch:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
RESUME_TRAIN_CHECKPOINT=/huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369/runs/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369/checkpoints/train_state_step006100.pt \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0 \
D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all SCRATCH_MODE=none \
HOLES_MIN=51 HOLES_MAX=64 EVAL_HOLES=60 EVAL_HOLES_LIST=53,60 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=7000 FULL_EVAL_N=512 FULL_ROLLOUT_KS="" \
EVAL_CHECKPOINT_STEPS=6300,6600,7000 EVAL_CHECKPOINT_HOLES_LIST=53,60 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 CASE_BANK_HOLES=53,60 CASE_BANK_N=4 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,3,5 \
RUN_NAME=gdn-transition-mixedhard-d224l12-step7000-20260703T<utc>-<sha> ./run.sh full
```

## 6. Artifacts

Pending.

## 7. Results

Pending.

Primary readouts:

- holes53 loop1/3/5 exact and blank accuracy.
- holes60 loop1/3/5 exact and blank accuracy.
- official blank ranges `46-50`, `51-55`, `56-64`.
- loop gain and hard-case wrong-cell trajectories.

## 8. Conclusions

Pending.

## 9. Submission Record

No submission/tag planned unless the score is unexpectedly strong and the mechanism conclusion remains clean.
