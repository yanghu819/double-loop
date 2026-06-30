# RWKV Official Sudoku Mask-Cliff Scale Gate

## 1. Metainfo

- Plan ID: `P-DIAG-002`
- Status: done
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-06-30 20:31:29 +0800`
- Execution window: `2026-06-30T12:32Z` to `2026-06-30T13:19Z`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Detached worktree:
  `/huyang2/double-loop/.worktrees/rwkv-maskcliff-83ccd29-20260630T1231Z`
- Git SHA: `83ccd29bb7c51a01f88fbf75bc79cbecd1d7c14a`

## 2. Hypothesis

We need to separate two explanations for the current official Sudoku plateau:

- GDN replacement is the regression: the original full RWKV+native FutureSeed
  backbone should beat the GDN plateau on the same official arrays.
- High blank-count official boards are the cliff: lower-blank buckets should
  open more strongly than the hardest bucket even under the RWKV backbone.

This is one diagnostic run, not a sweep. If it does not answer this split, the
next step should be a different mechanism question rather than another nearby
size/seed/loss table.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: full standalone RWKV + native terminal-state FutureSeed
- Architecture: `D_MODEL=256`, `LAYERS=12`, `HEADS=16`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=8`, final-loop loss
- Train: `FULL_STEPS=2000`, `FULL_BATCH=96`, `BLANK_LOSS_WEIGHT=8`
- Eval: official test split, `FULL_EVAL_N=512`
- Official blank-range eval: `46-50,51-55,56-64`
- Case-bank visualization: per blank-range groups, loops `1,2,3,5,8`
- CUDA path: `RWKV_KERNEL=statepassing`, `FORWARD_DTYPE=bfloat16`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule or postprocessing, no no-FS/GDN table in this run.

## 4. Environment

- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- Worktree:
  `/huyang2/double-loop/.worktrees/rwkv-maskcliff-83ccd29-20260630T1231Z`
- Caches/artifacts/runs/models under `/huyang2/double-loop`
- Python: `/opt/conda/bin/python`
- Torch: `2.7.0+cu126`
- GPU: `NVIDIA A100-SXM4-80GB`
- Peak CUDA allocated/reserved: `65664 MB / 70844 MB`
- Remote post-run check: GPU memory/utilization returned to `0 MiB / 0%`,
  `/tmp/aistation-ssh-password` absent.

## 5. Commands

Local static validation only:

```bash
python -m py_compile experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py
```

Remote launch shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
PIP_CACHE_DIR=/huyang2/double-loop/.cache/pip \
HF_HOME=/huyang2/double-loop/.cache/huggingface \
TORCH_HOME=/huyang2/double-loop/.cache/torch \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 BACKBONE=rwkv RWKV_KERNEL=statepassing \
D_MODEL=256 LAYERS=12 HEADS=16 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=8 FULL_STEPS=2000 FULL_BATCH=96 FULL_EVAL_N=512 \
HOLES_MIN=46 HOLES_MAX=64 EVAL_HOLES=56 EVAL_HOLES_LIST=56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=500,1000,1500,2000 \
SAVE_TRAIN_CHECKPOINT_EVERY=500 \
TRAIN_CHECKPOINT_DIR=/huyang2/double-loop/models/<run_name> \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=2 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,2,3,5,8 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- If step100 takes more than 15 minutes, stop by exact PID and relaunch the same
  diagnostic at `FULL_BATCH=64`, not a new idea.
- If CUDA OOMs, relaunch once at `D_MODEL=192/LAYERS=12` only if the OOM happens
  before meaningful training starts.
- If step500 CE is clearly worse than the historical D192 RWKV/GDN openings and
  blank accuracy remains near the no-FS plateau, stop and classify this as not a
  useful RWKV scale continuation.

Actual run name:

```bash
rwkv-official-maskcliff-d256l12-s2000-20260630T1240Z-83ccd29
```

## 6. Artifacts

Remote:

- Run dir:
  `/huyang2/double-loop/.worktrees/rwkv-maskcliff-83ccd29-20260630T1231Z/runs/rwkv-official-maskcliff-d256l12-s2000-20260630T1240Z-83ccd29`
- Launch:
  `/huyang2/double-loop/artifacts/launch/rwkv-official-maskcliff-d256l12-s2000-20260630T1240Z-83ccd29`
- Checkpoints, not committed:
  `/huyang2/double-loop/models/rwkv-official-maskcliff-d256l12-s2000-20260630T1240Z-83ccd29/train_state_step*.pt`

Committed archive:

- Root:
  `research/reports/visualizations/rwkv-official-maskcliff-scale-20260630/`
- Dashboard:
  `research/reports/visualizations/rwkv-official-maskcliff-scale-20260630/visualizations/index.html`
- Main metrics:
  `research/reports/visualizations/rwkv-official-maskcliff-scale-20260630/futureseed_loop_seed52.json`
- Case banks:
  `research/reports/visualizations/rwkv-official-maskcliff-scale-20260630/case_bank/official_b46_50/index.html`
  `research/reports/visualizations/rwkv-official-maskcliff-scale-20260630/case_bank/official_b51_55/index.html`
  `research/reports/visualizations/rwkv-official-maskcliff-scale-20260630/case_bank/official_b56_64/index.html`
- Source provenance:
  `source_HEAD.txt`, `source.patch`, `source_snapshot.tar.gz`

## 7. Results

Training and checkpoint eval:

| step | train CE | loop8 exact | loop8 blank_acc |
|---:|---:|---:|---:|
| 500 | 1.1107 | 0.0039 | 0.4758 |
| 1000 | 1.0397 | 0.0156 | 0.5020 |
| 1500 | 1.0157 | 0.0156 | 0.5089 |
| 2000 | 0.9882 | 0.0156 | 0.5190 |

Final mixed official test subset, `eval_n=512`:

| loop | exact | blank_acc |
|---:|---:|---:|
| 1 | 0.0000 | 0.4408 |
| 2 | 0.0234 | 0.4976 |
| 3 | 0.0234 | 0.5285 |
| 8 | 0.0234 | 0.5310 |

Official blank-range eval, `n=512` per bucket:

| blank range | loop8 exact | loop8 blank_acc | meaning |
|---|---:|---:|---|
| 46-50 | 0.9883 | 0.9993 | essentially solved |
| 51-55 | 0.0000 | 0.5432 | cliff: soft accuracy but no full-board exact |
| 56-64 | 0.0000 | 0.4893 | harder cliff |

Case-bank summary:

| bucket | exact | blank_acc | selected cases |
|---|---:|---:|---|
| official_b46_50 | 0.9922 | 0.9998 | solved_by_loop=2, almost_solved=2, hard_failure=0 |
| official_b51_55 | 0.0000 | 0.5326 | hard_failure=2 |
| official_b56_64 | 0.0000 | 0.4898 | hard_failure=2 |

## 8. Conclusions

High-information result.

This run says the current failure is not simply "GDN broke FutureSeed". Full
RWKV+native FutureSeed at larger D256/L12/loop8 can almost perfectly solve the
lower official blank bucket: `46-50` blanks reaches `0.9883` exact and `0.9993`
blank accuracy. So the original backbone still works when the mask is not too
deep.

The hard failure is a sharp mask/full-board exact cliff. Moving from `46-50` to
`51-55` blanks turns exact from `0.9883` to `0.0000`, even though blank accuracy
is still `0.5432`. This is not a subtle GDN-vs-RWKV difference; it is a global
consistency problem triggered by slightly deeper masks.

Loop is useful but front-loaded. On the mixed official subset, loop1 exact is
`0`, loop2 jumps to `0.0234`, and loops 3-8 mostly improve blank accuracy
without improving exact. In the solved lower bucket, loop2 already gives
`0.9297` exact and loop3 gives `0.9844`; in hard buckets, later loops do not
turn partial cell accuracy into board-level solutions.

Decision:

- Do not treat the D192/D256 GDN plateau as proof that FutureSeed failed.
- Do not keep blind same-shape long training as the main move; step1000-2000
  raised blank accuracy but exact was flat on the hard mixed checkpoint eval.
- The next high-ROI direction should focus on the `51-55` transition. It is the
  smallest hard bucket where the method falls from solved to zero, so it is the
  right proxy for discovering the missing global-consistency mechanism.
- Still no selector/search/repair/Sudoku rule. The bitter-lesson-friendly axis
  is a generic state/update/training formulation that makes loop continue
  resolving global constraints after it has high local blank accuracy.

## 9. Submission Record

Not a submission run.
