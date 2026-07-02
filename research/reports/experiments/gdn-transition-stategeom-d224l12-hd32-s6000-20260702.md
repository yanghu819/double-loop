# GDN FutureSeed D224/L12 State Geometry Scale

## 1. Metainfo

- Plan ID: `P-DIAG-014`
- Status: planned
- Local branch: `codex/gpu1-experiment-tracking`
- Planned time: `2026-07-02 19:00 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: pending plan commit

## 2. Hypothesis

P-DIAG-012 proved that D224/L12 with native FutureSeed and enough hard-stage
training can turn the `51-55` cliff into board-level exact. P-DIAG-013 then
showed that simply continuing the same state shape into a `56-64` hard tail did
not improve holes60 at step8000.

The next clean scaling question is whether the bottleneck is recurrent state
geometry. Current D224/L12 uses `HEADS=14`, `HEAD_DIM=16`, `GDN_EXPAND_V=4`,
so each layer has state size `14 * 64 * 16 = 14336`. Keeping token width fixed
but switching to `HEADS=7`, `HEAD_DIM=32` changes that to
`7 * 128 * 32 = 28672`. This doubles generic recurrent memory/state capacity
without adding Sudoku rules, repair, selector, or a hand objective.

Prediction:

- If state geometry is the bottleneck, the H7/D32 run should beat the H14/D16
  D224 baseline at the same step count, especially on holes53/holes60 loop5.
- If it only slows training or stays below the H14/D16 curve, the blocker is
  not simple state-matrix capacity. Then the next direction should be a cleaner
  recurrent state formulation, not another width/state table.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=224`, `LAYERS=12`, `HEADS=7`, `HEAD_DIM=32`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`
- Loop supervision: `LOOP_LOSS=all`
- Curriculum: `HOLE_STAGES=46-50:100,51-55:5900`
- Train: target total `FULL_STEPS=6000`, microbatch `FULL_BATCH=32`,
  `GRAD_ACCUM_STEPS=4`, effective batch `128`
- Eval: official test split, `FULL_EVAL_N=512`
- Checkpoint eval: steps `1000,3000,4500,6000`, holes `53,60`
- Official blank-range eval: `46-50,51-55,56-64`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`, no activation checkpoint
- Optimization: `LR=0.0015`, `WEIGHT_DECAY=0.001`, `BLANK_LOSS_WEIGHT=8`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule, no seed/LR/loss/gate table.

## 4. Environment

Pending GPU1 restart. If AIStation queues, wait for GPU1; do not use GPU2.

## 5. Commands

Remote launch shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=224 LAYERS=12 HEADS=7 HEAD_DIM=32 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 LOOP_LOSS=all \
HOLE_STAGES=46-50:100,51-55:5900 \
FULL_STEPS=6000 FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_EVAL_N=512 \
EVAL_HOLES=53 EVAL_HOLES_LIST=53,60 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=1000,3000,4500,6000 \
EVAL_CHECKPOINT_HOLES_LIST=53,60 \
SAVE_TRAIN_CHECKPOINT_EVERY=500 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 LR=0.0015 WEIGHT_DECAY=0.001 \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=3 CASE_BANK_EVAL_N=192 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- Stop exact PID on OOM or NaN.
- If step100 takes more than 15 minutes, stop and record throughput boundary.
- If step1000 is clearly below the H14/D16 curve and has no plausible catch-up
  signal, stop instead of burning the lease.
- If GPU1 lease is too short for step6000, prioritize checkpoint eval and
  metadata over waiting for disconnect.

Success criteria:

- Minimum: step6000 holes53 loop5 exact beats the D224/H14/D16 baseline
  `0.1680`, or holes60 loop5 exact reaches `>=0.20`.
- Strong: official `56-64` loop5 exact reaches `>=0.15` without hurting
  `46-50` and `51-55`.
- Failure: lower throughput plus no quality gain means state geometry alone is
  not the missing piece.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not a submission run.
