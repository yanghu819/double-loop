# gdn-transition-bigbatch-d224l12-step12600-20260707

## 1. Metainfo

- Plan ID: `P-DIAG-026`
- Status: discarded / early-stopped
- Run name: `gdn-transition-bigbatch-d224l12-step12600-20260707T0656Z-8487bdd`
- Branch: `codex/gpu1-experiment-tracking`
- Source SHA: `8487bdd0318b78016f5974f09831abcd7a458308`
- Machine: AIStation `GPU1` only
- Start: 2026-07-07 06:56 UTC

## 2. Hypothesis

P-DIAG-021 showed the strongest clean native FutureSeed GDN result so far:
loop5 full exact `0.2363`, loop1 to loop5 exact gain `+0.2129`, official
`51-55` exact `0.3555`, and `56-64` exact `0.1582`. The weak point is not
whether FutureSeed and loop compute work; they do. The weak point is that hard
holes60/64 checkpoint exact oscillates, which makes the hard-range upper bound
hard to trust.

P-DIAG-025 ruled out a small learned overwrite-rate gate: it was active, but did
not stabilize or improve the hard range. This run tests a simpler scaling
explanation: maybe each optimizer update sees too few hard boards, so board-level
consistency gradients are noisy. If true, a larger effective batch should make
holes60/64 smoother and preserve or improve hard exact without changing model
semantics.

## 3. Configuration

- Base checkpoint:
  `/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/checkpoints/train_state_step012000.pt`
- Data:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000`
- Sudoku: official 9x9, random holes, train split `train`, eval split `test`
- Backbone: native FutureSeed GDN
- Model: `D_MODEL=224`, `LAYERS=12`, `HEADS=14`, `HEAD_DIM=16`
- State: `GDN_EXPAND_V=4.0`, `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`
- Loops: `MAX_LOOPS=5`, `LOOP_LOSS=all`, fixed loop update
- Batch: microbatch `FULL_BATCH=32`, `GRAD_ACCUM_STEPS=8`, effective batch `256`
- Train: resume from global step12000 to `FULL_STEPS=12600`
- Eval:
  - checkpoint eval steps `12300,12600`
  - checkpoint holes `53,60,64`
  - final official blank ranges `46-50,51-55,56-64`
  - case bank loops `1,3,5`

## 4. Environment

GPU1 was queued for about three hours after restart and temporarily appeared as
generic `GPU:1`. Launch was held until the row became `Running` with resource
`NVIDIA-A800-SXM4-80GB:1`.

- Host: `beuau517asnnk-0`
- GPU: `NVIDIA A800-SXM4-80GB`, 81920 MiB
- Python: `/opt/conda/bin/python`
- Torch: `2.7.0+cu126`
- CUDA: verified available
- Note: `/huyang2/double-loop/.venv/bin/python` exists but currently lacks
  torch after the AIStation restart, so the formal launch uses the container's
  existing conda Python rather than installing packages.

## 5. Commands

Planned remote launch:

```bash
export GITHUB_TOKEN="${GITHUB_TOKEN}"
export REPO_OWNER=yanghu819
export REPO_NAME=double-loop
cd /huyang2/double-loop
git fetch origin codex/gpu1-experiment-tracking
git fetch origin codex/gpu1-experiment-tracking
git worktree add --detach /huyang2/double-loop/.worktrees/pdiag026-bigbatch-8487bdd-20260707T0648Z 8487bdd0318b78016f5974f09831abcd7a458308
cd /huyang2/double-loop/.worktrees/pdiag026-bigbatch-8487bdd-20260707T0648Z

PYTHON_BIN=/opt/conda/bin/python \
SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean CUDA_VISIBLE_DEVICES=0 \
SUDOKU_SIZE=9 HOLE_PATTERN=random \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
OFFICIAL_SUDOKU_TRAIN_SPLIT=train OFFICIAL_SUDOKU_EVAL_SPLIT=test \
RESUME_TRAIN_CHECKPOINT=/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/checkpoints/train_state_step012000.pt \
D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_EXPAND_V=4.0 GDN_USE_SHORT_CONV=0 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all LOOP_UPDATE_MODE=fixed \
FULL_BATCH=32 GRAD_ACCUM_STEPS=8 FULL_STEPS=12600 FULL_EVAL_N=512 FULL_LOG_EVERY=100 \
HOLE_STAGES=46-50:100,51-55:5900,51-64:1000,51-55:600,51-64:800,56-64:300,51-64:500,51-55:300,56-64:500,51-64:600,51-55:300,56-64:500,51-64:600,56-64:300,51-64:300 \
EVAL_HOLES=53 EVAL_HOLES_LIST=53,60,64 \
EVAL_CHECKPOINT_STEPS=12300,12600 EVAL_CHECKPOINT_HOLES_LIST=53,60,64 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
CASE_BANK_HOLES=official_b46_50:46-50,official_b51_55:51-55,official_b56_64:56-64 \
CASE_BANK_N=4 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,3,5 \
FULL_ROLLOUT_KS= RUN_NAME=gdn-transition-bigbatch-d224l12-step12600-20260707T0656Z-8487bdd \
./run.sh full
```

Kill criteria:

- Do not launch unless GPU1 is verified as A800.
- If step100 throughput makes step12600 infeasible within the lease, stop by exact
  PID and write `abort.json`.
- If checkpoint holes60/64 are clearly below P-DIAG-021/P-DIAG-025 neighboring
  curves, stop and classify batch256 as low ROI.

## 6. Artifacts

- Remote worktree:
  `/huyang2/double-loop/.worktrees/pdiag026-bigbatch-8487bdd-20260707T0648Z`
- Remote run dir:
  `/huyang2/double-loop/.worktrees/pdiag026-bigbatch-8487bdd-20260707T0648Z/runs/gdn-transition-bigbatch-d224l12-step12600-20260707T0656Z-8487bdd`
- Launch dir:
  `/huyang2/double-loop/artifacts/launch/gdn-transition-bigbatch-d224l12-step12600-20260707T0656Z-8487bdd`
- Initial failed launch:
  `gdn-transition-bigbatch-d224l12-step12600-20260707T0648Z-8487bdd` exited before training with `uv is not available; run ./setup.sh first.`
- Stopped process:
  - `bash ./run.sh full` PIDs `448`, `481`
  - `/opt/conda/bin/python study_rwkv_futureseed_loop.py` PID `483`
- Abort file:
  `/huyang2/double-loop/.worktrees/pdiag026-bigbatch-8487bdd-20260707T0648Z/runs/gdn-transition-bigbatch-d224l12-step12600-20260707T0656Z-8487bdd/abort.json`
- Remote metadata archive:
  `/huyang2/double-loop/artifacts/gdn-transition-bigbatch-d224l12-step12600-20260707T0656Z-8487bdd-metadata-light.tgz`
- Local metadata archive:
  `/Users/torusmini/Documents/double-loop-gpu1/.codex-transfer/gdn-transition-bigbatch-d224l12-step12600-20260707T0656Z-8487bdd-metadata-light.tgz`
- Local run artifacts:
  `/Users/torusmini/Documents/double-loop-gpu1/.codex-transfer/localrepo/runs/gdn-transition-bigbatch-d224l12-step12600-20260707T0656Z-8487bdd`

## 7. Results

Early checkpoint result at step12300:

| eval bucket | loop1 exact | loop5 exact | loop gain | loop5 blank |
|---|---:|---:|---:|---:|
| holes53 | 0.0137 | 0.1562 | +0.1426 | 0.5919 |
| holes60 | 0.0156 | 0.1660 | +0.1504 | 0.6124 |
| holes64 | 0.0176 | 0.1758 | +0.1582 | 0.5950 |

Comparison against the relevant neighbor:

| run | checkpoint | holes53 loop5 exact | holes60 loop5 exact | holes64 loop5 exact |
|---|---:|---:|---:|---:|
| P-DIAG-025 learned gate | 12600 | 0.1875 | 0.2246 | 0.2012 |
| P-DIAG-026 big batch | 12300 | 0.1562 | 0.1660 | 0.1758 |

P-DIAG-021 also had hard-range peaks around `holes60=0.2480` at step9000 and
`holes64=0.2441` at step11000. The big-batch run is well below that regime.

Throughput was acceptable: step12100 appeared around 6-7 minutes after launch,
and the run reached step12400 before the exact-PID stop. This was not a
throughput abort; it was a metric/ROI abort.

## 8. Conclusions

Decision: discard P-DIAG-026.

The useful signal is that loop computation is still real after the batch change:
loop1 to loop5 exact gain is about `+0.15` across holes53/60/64. But the whole
operating point shifted down. The hypothesis that the hard exact oscillation is
mainly optimizer noise from too-small effective batch is not supported.

The run was stopped by the predeclared checkpoint rule because holes60/64 were
clearly below P-DIAG-025 neighboring values and below P-DIAG-021 hard peaks.
Exact PIDs `483`, `481`, and `448` were killed; GPU1 returned to `0 MiB`.

Lesson: do not spend more budget on batch-size stability tables. The next useful
direction has to change the recurrent state information/content, improve data
scale/curriculum in a non-table way, or use checkpoint selection; simple
effective-batch doubling is not the stabilizer.

## 9. Submission Record

No submission. This is a mechanism/scaling diagnostic only.
