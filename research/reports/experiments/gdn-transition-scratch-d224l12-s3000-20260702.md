# GDN Transition Scratch D224/L12 S3000

## 1. Metainfo

- run_name: `gdn-transition-scratch-d224l12-s3000-20260702T1340Z`
- plan_id: `P-DIAG-016`
- machine: AIStation `GPU1` only
- local_start_time: `2026-07-02 21:40 CST`
- status: `in-progress`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

Raw D256 width, raw H7/D32 state geometry, and same-shape `56-64` hard-tail all failed as next scaling axes after D224/L12 opened `51-55`. The likely bottleneck is not just more channels; it is that later loops lack an independent generic work memory and tend to recompute the same state.

A gated scratch state is a minimal bitter-lesson-compatible modeling change: it gives each loop a learned, task-agnostic working state without Sudoku repair, search, selector, or hand rules. If this is the missing state/update capacity, scratch should become active and loop3->5 should keep improving hard boards earlier than the fixed D224 baseline.

## 3. Configuration

- Data: official EqR Sudoku arrays under `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000`
- Backbone: standalone native FutureSeed GDN, Triton recurrent kernel
- Model: `D_MODEL=224`, `LAYERS=12`, `HEADS=14`, `HEAD_DIM=16`, `CHANNEL_MULT=4`, `GDN_EXPAND_V=4.0`
- Loop: `MAX_LOOPS=5`, `L_CYCLES=2`, `LOOP_LOSS=all`
- Scratch: `SCRATCH_MODE=gated`, `SCRATCH_SCALE=1.0`, no scratch noise, no scratch Gaussian regularizer
- Batch: microbatch `32`, `GRAD_ACCUM_STEPS=4`, effective batch `128`
- Curriculum: `HOLE_STAGES=46-50:100,51-55:2900`
- Eval: checkpoint eval at `1000,2000,3000` on holes `53,60`; final official ranges `46-50,51-55,56-64`; case bank enabled
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair, no Sudoku-specific rule

## 4. Environment

To be filled after remote probe. Must verify:

- remote work_dir under `/huyang2/double-loop`
- detached source SHA
- CUDA-visible device is GPU1-local device `0`
- `.venv`, `.cache`, `artifacts`, `models`, `runs` stay under `/huyang2/double-loop`

## 5. Commands

Planned command shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
SUDOKU_SIZE=9 HOLE_PATTERN=random \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0 \
D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all \
SCRATCH_MODE=gated SCRATCH_SCALE=1.0 SCRATCH_NOISE_SCALE=0 SCRATCH_GAUSS_WEIGHT=0 \
HOLE_STAGES=46-50:100,51-55:2900 EVAL_HOLES=53 EVAL_HOLES_LIST=53,60 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=3000 FULL_EVAL_N=512 \
EVAL_CHECKPOINT_STEPS=1000,2000,3000 EVAL_CHECKPOINT_HOLES_LIST=53,60 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 CASE_BANK_N=6 CASE_BANK_EVAL_N=512 \
RUN_NAME=gdn-transition-scratch-d224l12-s3000-<timestamp>-<sha> ./run.sh full
```

Kill criteria:

- If step1000 holes53 loop5 exact/blank is not competitive with fixed D224 step1000 `0.0176/0.5284`, or holes60 loop5 is not competitive with `0.0215/0.5390`, stop.
- If `scratch_delta` is effectively zero at step1000, stop: the mechanism is not being used.
- If memory or speed is materially worse than fixed D224 with no metric signal, stop.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

No submission/tag planned unless score is meaningfully strong and mechanism evidence is clean.
