# GDN Transition Depth Scale D224/L16 Step4000

## 1. Metainfo

- run_name: `gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb`
- plan_id: `P-DIAG-022`
- machine: AIStation `GPU1` only
- local_start_time: `2026-07-04 16:41 CST`
- status: `failed`
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
- Actual execution note: the first `s4000` launch was interrupted by AIStation lease halt at step400 before the first saved train checkpoint. It was recorded with `abort.json`, then replaced by a shorter `s1000-restart1` gate using the same D224/L16 model, `SAVE_TRAIN_CHECKPOINT_EVERY=100`, `HOLE_STAGES=46-50:100,51-55:900`, and checkpoint eval `500,1000`.

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
- Partial abort file:
  `runs/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb/abort.json`
- Restart run:
  `gdn-transition-depthscale-d224l16-s1000-restart1-20260704T0924Z-74096fb`
- Restart remote run directory:
  `/huyang2/double-loop/.worktrees/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb/runs/gdn-transition-depthscale-d224l16-s1000-restart1-20260704T0924Z-74096fb`
- Restart local archive:
  `runs/gdn-transition-depthscale-d224l16-s1000-restart1-20260704T0924Z-74096fb`
- Restart visualization:
  `runs/gdn-transition-depthscale-d224l16-s1000-restart1-20260704T0924Z-74096fb/visualizations/index.html`

## 7. Results

First `s4000` launch:

- Interrupted by AIStation lease halt at step400 before checkpoint step500.
- Last log: step400 CE `1.2457`.
- No train checkpoint was available, so this leg was not resumed.

Restart `s1000` gate:

| Checkpoint | Train CE | holes53 loop5 exact / blank | holes60 loop5 exact / blank | holes64 loop5 exact / blank |
|---:|---:|---:|---:|---:|
| 500 | `1.1584` | `0.0000 / 0.4480` | `0.0000 / 0.4589` | `0.0000 / 0.4568` |
| 1000 | `0.9999` | `0.0176 / 0.5104` | `0.0215 / 0.5174` | `0.0254 / 0.5141` |

Final full-board eval at step1000:

| Loop | exact | blank_acc |
|---:|---:|---:|
| 1 | `0.0195` | `0.4986` |
| 2 | `0.0234` | `0.5172` |
| 3 | `0.0234` | `0.5195` |
| 4 | `0.0234` | `0.5195` |
| 5 | `0.0234` | `0.5192` |

Official blank-range eval at loop5:

| Blank range | exact | blank_acc |
|---|---:|---:|
| `46-50` | `0.9922` | `0.9995` |
| `51-55` | `0.0000` | `0.5445` |
| `56-64` | `0.0000` | `0.4836` |

Case bank:

| Bank | exact | blank_acc | selected |
|---|---:|---:|---|
| `official_b46_50` | `0.9766` | `0.9990` | 4 solved-by-loop, 4 almost-solved |
| `official_b51_55` | `0.0000` | `0.5354` | 4 hard-failure |
| `official_b56_64` | `0.0000` | `0.4870` | 4 hard-failure |

## 8. Conclusions

Decision: mark P-DIAG-022 as `failed` / low ROI depth-scaling boundary.

The model is trainable and fits GPU1, but it is a bad scaling axis at this point. D224/L16 uses about `63.7GB` and takes about `3916s` to reach step1000. At that point it only reaches loop5 exact/blank `0.0234/0.5192`, with official `51-55` and `56-64` exact still `0`. Compared with the cheaper D224/L12 early curve, the exact is only tied-ish while blank accuracy and throughput are worse.

The important lesson is not "scale is wrong"; it is that naive extra depth is the wrong kind of scale for the current bottleneck. The model still learns local blanks and loop gives small improvement, but deeper same-family computation does not convert local blank accuracy into global consistency. Do not run L14/L18/L20 or longer L16 from scratch. The next useful scaling axis should be data/curriculum with demonstrated slope, or a more efficient recurrent state/update formulation that changes how global consistency is represented.

## 9. Submission Record

No tag unless the primary score is meaningfully strong and mechanism conclusion is clean.
