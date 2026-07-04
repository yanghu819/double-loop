# GDN Transition Cyclic Bridge Loop8 D224/L12 Step12000

## 1. Metainfo

- run_name: `gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z`
- plan_id: `P-DIAG-020`
- machine: AIStation `GPU1` only
- local_start_time: `2026-07-03 15:57 CST`
- status: `failed`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

P-DIAG-019 gave the cleanest current lesson: a staged bridge can push the hardest official blank range, but a long pure `56-64` tail damages mixed quality. The next scaling question is whether a cyclic bridge-hard distribution plus more recurrent compute can move the boundary again without losing the already-open `51-55` transition.

This is a bitter-lesson run: more compute, more effective batch, more loop budget, and a broader data distribution. No Sudoku repair, no selector, no oracle rollout, no search, no task-specific rule.

## 3. Configuration

- Resume checkpoint: P-DIAG-019 best mixed-hard checkpoint `train_state_step008400.pt`.
- Data: official EqR Sudoku arrays.
- Backbone: native FutureSeed GDN, D224/L12/H14/D16, `GDN_EXPAND_V=4.0`.
- Loop: `MAX_LOOPS=8`, `LOOP_LOSS=all`.
- Effective batch: microbatch `32`, grad accumulation `8`, effective batch `256`.
- Active continuation after step8400:
  - `56-64:300`
  - `51-64:500`
  - `51-55:300`
  - `56-64:500`
  - `51-64:600`
  - `51-55:300`
  - `56-64:500`
  - `51-64:600`
- Target: `FULL_STEPS=12000`, checkpoint eval at `9000,10000,11000,12000`.
- No scratch, no extra loss, no repair/search/selector, no CPU smoke, no GPU2.

## 4. Prediction And Kill Criteria

Prediction:

- If the main bottleneck is data continuity plus recurrent compute, loop8 should improve `56-64` exact above P-DIAG-019 final official `0.1250`, ideally `>=0.18`, while keeping `51-55 >=0.25`.
- If loop5 already saturates the useful recurrent computation, loop8 will cost more but produce the same checkpoint readout.

Kill criteria:

- Stop if NaN/OOM.
- Stop if throughput is so slow that step9000 would take unreasonably long relative to the remaining GPU lease.
- Discard if final official `51-55 <0.25` or official `56-64 <=0.1250`.
- Discard if loop8 does not improve over loop5/loop1 on exact/blank and only repeats the same operating point.

## 5. Commands

Planned remote launch:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
SUDOKU_SIZE=9 PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
RESUME_TRAIN_CHECKPOINT=/huyang2/double-loop/.worktrees/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/runs/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/checkpoints/train_state_step008400.pt \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0 \
D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=8 LOOP_LOSS=all SCRATCH_MODE=none \
HOLE_STAGES=46-50:100,51-55:5900,51-64:1000,51-55:600,51-64:800,56-64:300,51-64:500,51-55:300,56-64:500,51-64:600,51-55:300,56-64:500,51-64:600 \
EVAL_HOLES=60 EVAL_HOLES_LIST=53,60,64 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=8 FULL_STEPS=12000 FULL_EVAL_N=512 \
FULL_ROLLOUT_KS="" FULL_LOG_EVERY=100 \
EVAL_CHECKPOINT_STEPS=9000,10000,11000,12000 EVAL_CHECKPOINT_HOLES_LIST=53,60,64 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
CASE_BANK_HOLES=53,60,64 CASE_BANK_N=4 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,3,5,8 \
RUN_NAME=gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z ./run.sh full
```

## 6. Artifacts

- Remote worktree: `/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z`
- First launch dir: `/huyang2/double-loop/artifacts/launch/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z`
- First partial run dir: `/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z`
- Resume100 launch dir: `/huyang2/double-loop/artifacts/launch/gdn-transition-cyclicbridge-loop8-d224l12-step12000-resume100-20260704T0008Z-3f09ea5`
- Resume100 partial run dir: `/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop8-d224l12-step12000-resume100-20260704T0008Z-3f09ea5`
- Abort file: `runs/gdn-transition-cyclicbridge-loop8-d224l12-step12000-resume100-20260704T0008Z-3f09ea5/abort.json`

## 7. Results

The original loop8/effective-batch256 launch resumed correctly from step8400 and reached:

| step | stage | CE | total | loop1 | loop_last |
|---:|---|---:|---:|---:|---:|
| 8500 | 56-64 | `0.5318` | `0.6275` | `1.0331` | `0.5318` |
| 8600 | 56-64 | `0.4475` | `0.5686` | `1.0671` | `0.4475` |

AIStation halted before the first planned eval/checkpoint at step9000. A resume100 run was launched with `SAVE_TRAIN_CHECKPOINT_EVERY=100`, but it was killed after about eight minutes because it still had not reached the first 100-step log while using about `76GB` GPU memory.

No score/eval result was produced. This is a throughput failure, not evidence against cyclic bridge-hard data.

## 8. Conclusions

Do not continue loop8 plus effective batch256 on the current D224/L12 GDN formulation. It is the wrong scaling axis: memory is nearly full, throughput is too low, and the information gain per GPU hour is worse than P-DIAG-019.

The useful next move is P-DIAG-021: keep the proven loop5/effective batch128 path, keep cyclic bridge-hard data, train longer, and save train checkpoints every 100 steps so AIStation interruptions do not erase progress.

## 9. Submission Record

No tag unless score is meaningfully strong and mechanism conclusion is clean.
