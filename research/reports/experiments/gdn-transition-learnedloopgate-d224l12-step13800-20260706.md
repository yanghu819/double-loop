# GDN Transition Learned Loop Gate D224/L12 Step13800

## 1. Metainfo

- run_name: `gdn-transition-learnedloopgate-d224l12-step13800-20260706T2235Z-23090be-a800`
- plan_id: `P-DIAG-025`
- machine: AIStation `GPU1` only
- local_prepared_time: `2026-07-06 16:01 CST`
- launched_time: `2026-07-06 22:35 CST`
- status: `discarded`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

P-DIAG-021 is the current strongest clean result: D224/L12 native FutureSeed GDN plus cyclic bridge-hard training reaches loop5 exact `0.2363`, with loop1->5 exact gain `+0.2129`. That proves loop computation is real.

The remaining weakness is stability. Holes60 and holes64 checkpoint exact oscillate instead of climbing monotonically. Continuing the identical fixed-update training risks just sampling another high or low point.

This probe asks a sharper mechanism question: is the fixed recurrent update rate too rigid? A learned per-loop/per-stream update gate lets the model decide how much to revise H/L state at each loop, without Sudoku rules, repair, search, selector, or any hand path logic. It is a generic state-dynamics change, not a task hack.

## 3. Configuration

- Source SHA: `23090befdaff42a353ca282fdfe083120999b2b3`.
- Resume checkpoint: P-DIAG-021 `train_state_step012000.pt`, verified on GPU1 before launch.
- Data: official EqR Sudoku arrays.
- Backbone: native FutureSeed GDN, D224/L12/H14/D16, `GDN_EXPAND_V=4.0`.
- Loop: `MAX_LOOPS=5`, `LOOP_LOSS=all`.
- New mechanism: `LOOP_UPDATE_MODE=learned_gate`, `LOOP_UPDATE_GATE_INIT=0.95`.
- Effective batch: microbatch `32`, grad accumulation `4`, effective batch `128`.
- Active continuation after step12000:
  - `51-64:600`
  - `56-64:600`
  - `51-64:600`
- Target: `FULL_STEPS=13800`.
- Checkpoint eval: `12600,13200,13800` on holes `53,60,64`.
- Train checkpoint safety: `SAVE_TRAIN_CHECKPOINT_EVERY=100`.
- Final official blank ranges: `46-50,51-55,56-64`.
- Case-bank loops: `1,3,5`.

## 4. Prediction And Kill Criteria

Prediction:

- If fixed update rate is part of the oscillation, learned gates should move away from exactly `0.95` and either stabilize or improve holes60/64 exact.
- A strong positive is holes60 or holes64 loop5 exact beating the P-DIAG-021 peak, roughly `>=0.25`, while preserving official `51-55 >=0.33`.
- A weak positive is similar final score but lower checkpoint oscillation and preserved loop1->5 gain.

Kill criteria:

- Stop if step12600 holes53/60/64 all trail P-DIAG-021 step12000 by more than `0.03` exact and blank accuracy also drops.
- Stop on NaN/OOM, optimizer-state load failure, or learned gates collapsing to unusable values.
- Discard if gates move but only reduce blank accuracy/exact, or if final hard exact does not beat the prior boundary.

## 5. Commands

Actual remote launch from detached worktree:

```bash
cd /huyang2/double-loop/.worktrees/pdiag025-learnedloopgate-23090be-20260706T2120
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
SUDOKU_SIZE=9 PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
RESUME_TRAIN_CHECKPOINT=/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/checkpoints/train_state_step012000.pt \
SAVE_TRAIN_CHECKPOINT_EVERY=100 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0 \
D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all SCRATCH_MODE=none \
LOOP_UPDATE_MODE=learned_gate LOOP_UPDATE_GATE_INIT=0.95 \
HOLE_STAGES=46-50:100,51-55:5900,51-64:1000,51-55:600,51-64:800,56-64:300,51-64:500,51-55:300,56-64:500,51-64:600,51-55:300,56-64:500,51-64:600,51-64:600,56-64:600,51-64:600 \
EVAL_HOLES=60 EVAL_HOLES_LIST=53,60,64 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=13800 FULL_EVAL_N=512 \
FULL_ROLLOUT_KS="" FULL_LOG_EVERY=100 \
EVAL_CHECKPOINT_STEPS=12600,13200,13800 EVAL_CHECKPOINT_HOLES_LIST=53,60,64 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
CASE_BANK_HOLES=53,60,64 CASE_BANK_N=4 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,3,5 \
RUN_NAME=gdn-transition-learnedloopgate-d224l12-step13800-20260706T2235Z-23090be-a800 ./run.sh full
```

Earlier launch scripts with the same config were created for `2125Z`, `2140Z`, `2155Z`, and `2210Z`; they are archived as platform-aborted attempts rather than separate modeling runs.

## 6. Artifacts

- Local record: `research/reports/experiments/gdn-transition-learnedloopgate-d224l12-step13800-20260706.md`
- Remote worktree: `/huyang2/double-loop/.worktrees/pdiag025-learnedloopgate-23090be-20260706T2120`
- Remote run dir: `/huyang2/double-loop/.worktrees/pdiag025-learnedloopgate-23090be-20260706T2120/runs/gdn-transition-learnedloopgate-d224l12-step13800-20260706T2235Z-23090be-a800`
- Launch script: `/huyang2/double-loop/.worktrees/pdiag025-learnedloopgate-23090be-20260706T2120/artifacts/launch/gdn-transition-learnedloopgate-d224l12-step13800-20260706T2235Z-23090be-a800.sh`
- Light metadata archive: remote `/huyang2/double-loop/artifacts/gdn-transition-learnedloopgate-d224l12-step13800-20260706T2235Z-23090be-a800-metadata-light.tgz`; local `.codex-transfer/gdn-transition-learnedloopgate-d224l12-step13800-20260706T2235Z-23090be-a800-metadata-light.tgz`. This excludes checkpoints and `source_snapshot.tar.gz`.
- Initial process: `bash ./run.sh full` PID `145`/`180`; Python training PID `182`.
- GPU prelaunch check: `NVIDIA A800-SXM4-80GB`, `0 MiB / 81920 MiB`, util `0%`.
- Checkpoint verified: `/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/checkpoints/train_state_step012000.pt`, `161M`.
- Access note: AIStation repeatedly bounced GPU1 between `Pause`, `Pending`, `ImagePulling`, and `Running`; Kimi WebBridge was used only to restore GPU1, and launch used API/SSH after verifying the row name.
- GPU1 restore note: the failed attempts correlated with GPU1 being restored as generic `GPU:1`. The final active launch was started only after the restore dialog was verified to contain environment `GPU1`, resource group `8A100_80_dev`, CPU/GPU `16/1`, and accelerator type `NVIDIA-A800-SXM4-80GB`; `probe GPU1` then reported `NVIDIA A800-SXM4-80GB, 0 MiB / 81920 MiB`.
- Aborted prelaunch attempts:
  - `gdn-transition-learnedloopgate-d224l12-step13800-20260706T2125Z-23090be` wrote only setup/resume log, was paused before first train step, and has `abort.json` with reason `AIStation paused after nohup launch before first train step; relaunched with setsid`.
  - `gdn-transition-learnedloopgate-d224l12-step13800-20260706T2140Z-23090be` also wrote only setup/resume log; no-start probe showed AIStation killed the background job after SSH disconnected.
  - `gdn-transition-learnedloopgate-d224l12-step13800-20260706T2155Z-23090be` ran foreground but the remote session closed before the first train log.
  - `gdn-transition-learnedloopgate-d224l12-step13800-20260706T2210Z-23090be` ran foreground with heartbeat but AIStation closed it before device/resume completed; `abort.json` records the generic `GPU:1` restore issue.
- Active launch note: `2235Z-23090be-a800` is the same modeling config, relaunched after GPU1 A800 restore and verified `nvidia-smi`.

## 7. Results

Completed. Initial monitor showed Python PID `182`, GPU memory about `48GB`, utilization about `80-90%`. GPU memory returned to `0 MiB` after completion.

First train readout after resume:

| step | stage | ce | total | loop1 | loop_last | upd_l | upd_h | checkpoint |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 12100 | 14:`51-64` | 0.0022 | 0.1626 | 0.7440 | 0.0022 | 0.952 | 0.952 | `train_state_step012100.pt` |

Checkpoint trajectory:

| Step | holes53 loop5 exact / blank | holes60 loop5 exact / blank | holes64 loop5 exact / blank | loop5 gate L/H |
|---:|---:|---:|---:|---:|
| 12600 | `0.1875 / 0.6011` | `0.2246 / 0.6310` | `0.2012 / 0.5993` | `0.9677 / 0.9655` |
| 13200 | `0.1680 / 0.5844` | `0.2051 / 0.6089` | `0.1973 / 0.5980` | `0.9703 / 0.9681` |
| 13800 | `0.2285 / 0.6133` | `0.2168 / 0.6281` | `0.2305 / 0.6127` | `0.9845 / 0.9818` |

Final full-board metrics:

| Loop | exact | blank_acc |
|---:|---:|---:|
| 1 | `0.0234` | `0.5346` |
| 2 | `0.0703` | `0.5808` |
| 3 | `0.1680` | `0.6018` |
| 4 | `0.2129` | `0.6133` |
| 5 | `0.2266` | `0.6166` |

Official blank-range metrics:

| Blank range | loop5 exact | loop5 blank_acc |
|---|---:|---:|
| `46-50` | `1.0000` | `1.0000` |
| `51-55` | `0.3496` | `0.7150` |
| `56-64` | `0.1348` | `0.5472` |

Case-bank artifacts:

- `official_b46_50`: exact `1.0000`, blank_acc `1.0000`, selected `{'solved_by_loop': 4, 'almost_solved': 0, 'hard_failure': 0}`.
- `official_b51_55`: exact `0.3398`, blank_acc `0.7111`, selected `{'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}`.
- `official_b56_64`: exact `0.1133`, blank_acc `0.5674`, selected `{'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}`.

## 8. Conclusions

Discard this mechanism as a score/stability improvement.

The gate did exactly what this probe asked mechanically: it moved from the `0.95` initialization to roughly `0.9845/0.9818` by the final loop. So this is not a wiring failure.

But it did not solve the real issue. P-DIAG-021 final full loop5 exact/blank was `0.2363/0.6287`; this run ends at `0.2266/0.6166`. The transition range is preserved, with official `51-55=0.3496` versus P-DIAG-021 `0.3555`, but the hardest official range regresses from `56-64=0.1582` to `0.1348`.

The useful insight is narrower: fixed update rate is not the main bottleneck. A learned gate can move and preserve loop gain, but under the same CE/all-loop pressure it tends to learn a more aggressive update schedule, not a stabilizing correction policy. The next high-ROI direction should not sweep gate init/bias/temperature. It should either return to clean scaling/curriculum with checkpoint selection, or use a qualitatively different generic recurrent state formulation that changes what information is stored, not just how much of the state is overwritten.

Operational lesson: the failed prelaunches were platform provenance, not model failures. GPU1 had been restored as generic `GPU:1`; the stable run started only after Kimi WebBridge restore selected `NVIDIA-A800-SXM4-80GB` and probe verified the actual A800.

## 9. Submission Record

Not applicable.
