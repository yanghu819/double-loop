# GDN Transition Learned Loop Gate D224/L12 Step13800

## 1. Metainfo

- run_name: `gdn-transition-learnedloopgate-d224l12-step13800-20260706T0801Z-6c72675`
- plan_id: `P-DIAG-025`
- machine: AIStation `GPU1` only
- local_prepared_time: `2026-07-06 16:01 CST`
- status: `approved, blocked before launch by AIStation/VPN timeout`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

P-DIAG-021 is the current strongest clean result: D224/L12 native FutureSeed GDN plus cyclic bridge-hard training reaches loop5 exact `0.2363`, with loop1->5 exact gain `+0.2129`. That proves loop computation is real.

The remaining weakness is stability. Holes60 and holes64 checkpoint exact oscillate instead of climbing monotonically. Continuing the identical fixed-update training risks just sampling another high or low point.

This probe asks a sharper mechanism question: is the fixed recurrent update rate too rigid? A learned per-loop/per-stream update gate lets the model decide how much to revise H/L state at each loop, without Sudoku rules, repair, search, selector, or any hand path logic. It is a generic state-dynamics change, not a task hack.

## 3. Configuration

- Source SHA: `6c72675aca4a06bd4f0333ef1f649434c39a2a10`.
- Resume checkpoint: P-DIAG-021 `train_state_step012000.pt`, to be verified on GPU1 before launch.
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

Planned remote launch after AIStation access is restored:

```bash
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
RUN_NAME=gdn-transition-learnedloopgate-d224l12-step13800-20260706T0801Z-6c72675 ./run.sh full
```

## 6. Artifacts

- Local record: `research/reports/experiments/gdn-transition-learnedloopgate-d224l12-step13800-20260706.md`
- Expected remote run dir: `/huyang2/double-loop/.worktrees/pdiag025-learnedloopgate-6c72675/runs/gdn-transition-learnedloopgate-d224l12-step13800-20260706T0801Z-6c72675`
- Prelaunch block:
  - AIStation API `status GPU1 GPU2` failed with `connect ETIMEDOUT 172.16.78.10:32206`.
  - Kimi WebBridge navigation to `https://172.16.78.10:32206/index.html#/developEnv/list` failed with `page load timeout (30s)`.
  - EasyConnect processes were running, but direct `curl -k -m 5` to the same host timed out.

## 7. Results

Not launched yet.

## 8. Conclusions

Pending. The chosen next experiment is not more noise, width, depth, or a seed table. It is a single generic state-dynamics probe from the strongest existing checkpoint.

If AIStation access remains unavailable, this plan should resume exactly from §5 after VPN/browser access is restored.

## 9. Submission Record

Not applicable.
