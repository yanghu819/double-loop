# GDN Transition Depth Scale D224/L16 Step4000

## 1. Metainfo

- run_name: `gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb`
- plan_id: `P-DIAG-022`
- machine: AIStation `GPU1` only
- local_start_time: `2026-07-04 16:41 CST`
- status: `running`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

P-DIAG-021 showed that native FutureSeed GDN plus loop5 is no longer toy: loop1 to loop5 gives about `+0.21` full-board exact, and cyclic bridge-hard training raises official `56-64` exact above the previous hard boundary. The remaining failure is not local token learning; it is stable full-board consistency under many blanks.

This experiment tests the clean scaling explanation: the D224/L12 recurrent state formulation may be too shallow to stabilize hard global consistency. Increasing depth to L16 is a generic model-capacity/state-computation scale axis. If depth is the missing axis, it should beat the D224/L12 same-step curve before we spend another long lease.

No Sudoku repair, no search, no selector, no oracle rollout, no task-specific rules.

## 3. Configuration

- Source SHA: `74096fb5e2ea299ea09fef71e0e8f2e36f5a9384`.
- Data: official EqR Sudoku arrays.
- Backbone: native FutureSeed GDN, D224/L16/H14/D16, `GDN_EXPAND_V=4.0`.
- Loop: `MAX_LOOPS=5`, `LOOP_LOSS=all`.
- Effective batch: microbatch `32`, grad accumulation `4`, effective batch `128`.
- Curriculum:
  - `46-50:100`
  - `51-55:2900`
  - `51-64:1000`
- Target: `FULL_STEPS=4000`.
- Checkpoint eval: `1000,2000,3000,4000` on holes `53,60,64`.
- Final official blank ranges: `46-50,51-55,56-64`.
- Case-bank loops: `1,3,5`.

## 4. Prediction And Kill Criteria

Prediction:

- Success: by step3000/4000, holes53 or holes60 loop5 exact clearly beats the D224/L12 same-step curve; useful threshold is official `51-55 >=0.08` or holes60 loop5 exact `>=0.15` by step4000.
- Strong success: depth improves hard exact without reducing loop gain; loop5 should still add real board-level exact over loop1.

Kill criteria:

- Stop on OOM/NaN.
- Stop at step1000 if it is clearly worse than D224/L12 step1000 and CE/blank slope is flat.
- Stop at step2000/3000 if it does not beat the D224/L12 curve enough to justify deeper compute.
- Do not turn this into L14/L18/L20 or LR/seed/loss tables.

## 5. Commands

Planned remote launch after prelaunch commit:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
SUDOKU_SIZE=9 PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SAVE_TRAIN_CHECKPOINT_EVERY=500 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0 \
D_MODEL=224 LAYERS=16 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all SCRATCH_MODE=none \
HOLE_STAGES=46-50:100,51-55:2900,51-64:1000 \
EVAL_HOLES=60 EVAL_HOLES_LIST=53,60,64 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=4000 FULL_EVAL_N=512 \
FULL_ROLLOUT_KS="" FULL_LOG_EVERY=100 \
EVAL_CHECKPOINT_STEPS=1000,2000,3000,4000 EVAL_CHECKPOINT_HOLES_LIST=53,60,64 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
CASE_BANK_HOLES=53,60,64 CASE_BANK_N=4 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,3,5 \
RUN_NAME=gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb ./run.sh full
```

## 6. Artifacts

- Remote worktree:
  `/huyang2/double-loop/.worktrees/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb`
- Remote run directory:
  `/huyang2/double-loop/.worktrees/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb/runs/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb`
- Launch script:
  `/huyang2/double-loop/artifacts/launch/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb.sh`
- Launcher log:
  `/huyang2/double-loop/artifacts/launch/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb.launcher.log`
- PID file:
  `/huyang2/double-loop/artifacts/launch/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb.pid`
- Current PID at launch: `2319`
- First status: fits GPU1 at about `63.7GB` used; step100 appeared at about `10.5m` with CE `1.8350`, no OOM/NaN.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

No tag unless the primary score is meaningfully strong and mechanism conclusion is clean.
