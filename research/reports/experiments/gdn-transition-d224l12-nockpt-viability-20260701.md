# GDN FutureSeed D224/L12 No-Checkpoint Viability

## 1. Metainfo

- Plan ID: `P-DIAG-010`
- Status: done
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-07-01 18:48:31 +0800`
- Run start: `2026-07-01 18:50 +0800`
- Run archived: `2026-07-01 19:04 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: `e97469d41fe88b4beddbfc8213e4c4151aa39d5a`
- Run name: `gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d`
- Remote run dir: `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-e97469d-20260701T1050Z/runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d`
- Local archive: `runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d`
- Parent evidence:
  - P-DIAG-006 D224/L12 with activation checkpoint reached NaN at step100.
  - P-DIAG-008 D192/L12 no-checkpoint stayed finite through step700.
  - P-DIAG-009 showed all-loop supervision makes intermediate states readable
    but does not solve the 51-55 cliff at D192.

## 2. Hypothesis

The next real scale axis is a larger model, not another loop-loss or objective
variant. The earlier D224/L12 failure is confounded by activation checkpointing.
If the bigger network is viable without checkpointing at a smaller microbatch,
then D224/L12 should at least fit, stay finite, and start learning the 51-55
stage.

Prediction:

- If D224/L12 no-checkpoint fits in 80GB and stays finite, the activation
  checkpoint/custom-backward interaction was the main D224 failure mode.
- If it OOMs or NaNs even without checkpointing, D224/L12 expand_v4 is currently
  outside the stable single-A800 scale envelope.

This is a short viability gate because GPU1 has about 20 minutes left. It is not
a score claim and not a sweep.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=224`, `LAYERS=12`, `HEADS=14`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`
- Loop supervision: `LOOP_LOSS=all`
- Curriculum: `HOLE_STAGES=46-50:100,51-55:200`
- Train: `FULL_STEPS=300`, microbatch `FULL_BATCH=32`,
  `GRAD_ACCUM_STEPS=4`, effective batch `128`
- Eval: official test split, `FULL_EVAL_N=512`
- Checkpoint eval: steps `100,300`, holes53
- Official blank-range eval: `46-50,51-55,56-64`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`, no activation checkpoint
- Optimization: `LR=0.0015`, `WEIGHT_DECAY=0.001`, `BLANK_LOSS_WEIGHT=8`
- Forbidden: no CPU smoke, no GPU2, no activation checkpoint, no
  selector/search/repair/oracle rollout, no Sudoku-specific rule, no
  LR/seed/loss-weight sweep.

## 4. Environment

- GPU: AIStation `GPU1`, A800 80GB
- CUDA visibility: `CUDA_VISIBLE_DEVICES=0`
- Python: `/opt/conda/bin/python`
- Torch: `2.7.0+cu126`
- Device: `cuda`
- GDN mode: `triton_recurrent`
- Forward dtype: `bfloat16`
- Activation checkpoint: disabled
- Git dirty at run launch: false
- Observed memory: about `44.8GB` during the short run

## 5. Commands

Remote launch shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 LOOP_LOSS=all \
HOLE_STAGES=46-50:100,51-55:200 \
FULL_STEPS=300 FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_EVAL_N=512 \
HOLES_MIN=46 HOLES_MAX=55 EVAL_HOLES=53 EVAL_HOLES_LIST=50,53,56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=100,300 EVAL_CHECKPOINT_HOLES_LIST=53 \
SAVE_TRAIN_CHECKPOINT_EVERY=100 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=50 \
BLANK_LOSS_WEIGHT=8 LR=0.0015 WEIGHT_DECAY=0.001 \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=2 CASE_BANK_EVAL_N=128 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- If CUDA OOMs, stop exact PID and archive.
- If step50 or step100 is NaN, stop exact PID and archive.
- If step100 takes too long to leave time for archive, stop and classify as
  throughput-infeasible under current lease.
- If GPU lease expires, restart/queue GPU1 and resume from the latest saved
  train checkpoint.

Success criteria:

- Minimal: step100 finite and checkpoint eval written.
- Strong viability: step300 finite, hard-stage CE decreases from step150/200 to
  step300, and memory stays below 80GB.
- Long-scale trigger: no OOM/NaN and checkpoint metrics are not clearly worse
  than D192/L12 all-loop at comparable early budget.

## 6. Artifacts

- `runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/config.json`
- `runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/logs/run.log`
- `runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/output/checkpoint_eval_step000100.json`
- `runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/output/checkpoint_eval_step000300.json`
- `runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/output/case_bank/`
- `runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/visualizations/index.html`
- `runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/abort.json`
- Remote train checkpoints were intentionally not committed:
  - `checkpoints/train_state_step000100.pt`
  - `checkpoints/train_state_step000200.pt`
  - `checkpoints/train_state_step000300.pt`

## 7. Results

Training finished the planned 300-step gate before the lease boundary. I still
wrote `abort.json` after stopping by exact known PIDs because the GPU lease was
under one minute and I did not want AIStation to hard-kill artifact collection.

Training CE:

| Step | Stage | CE | Total | Loop1 CE | Loop-last CE |
|---:|---|---:|---:|---:|---:|
| 50 | 46-50 | 1.7843 | 1.7836 | 1.7821 | 1.7843 |
| 100 | 46-50 | 1.4368 | 1.4357 | 1.4339 | 1.4368 |
| 150 | 51-55 | 1.4890 | 1.4889 | 1.4895 | 1.4890 |
| 200 | 51-55 | 1.2633 | 1.2633 | 1.2656 | 1.2633 |
| 250 | 51-55 | 1.1265 | 1.1290 | 1.1386 | 1.1265 |
| 300 | 51-55 | 1.0738 | 1.0768 | 1.0882 | 1.0738 |

Checkpoint eval on holes53:

| Checkpoint | Loop | Exact | Blank acc |
|---:|---:|---:|---:|
| 100 | 5 | 0.0000 | 0.2443 |
| 300 | 1 | 0.0000 | 0.4609 |
| 300 | 2 | 0.0000 | 0.4679 |
| 300 | 3 | 0.0000 | 0.4699 |
| 300 | 4 | 0.0000 | 0.4702 |
| 300 | 5 | 0.0020 | 0.4704 |

Final official blank-range eval:

| Blank range | Exact | Valid | Solved | Blank acc |
|---|---:|---:|---:|---:|
| 46-50 | 0.0449 | 0.0449 | 0.0449 | 0.9173 |
| 51-55 | 0.0000 | 0.0000 | 0.0000 | 0.4944 |
| 56-64 | 0.0000 | 0.0000 | 0.0000 | 0.4429 |

Loop gain at step300 exists only as soft accuracy:

- holes53 blank acc loop1 to loop5: `0.4609 -> 0.4704`
- holes53 exact loop1 to loop5: `0.0000 -> 0.0020`
- full-board eval loop1 to loop5 blank acc: `0.4623 -> 0.4717`
- full-board eval loop1 to loop5 exact: `0.0000 -> 0.0000`

## 8. Conclusions

This is a positive scaling-viability result, not a score result.

The important answer is that D224/L12 no-checkpoint is inside the 80GB envelope:
it fits, stays finite, and learns the hard `51-55` stage quickly. The prior
D224/L12 and D192/L12 NaNs are now much more likely to be an activation
checkpoint/custom GDN backward interaction than an inherent width/depth failure.

Quality is still early. At only 300 steps, `51-55` exact is still zero and
loop-depth mostly improves blank accuracy rather than full-board exact. But the
hard-stage CE slope is strong enough to justify one long D224/L12 no-checkpoint
resume from `train_state_step000300.pt` after GPU1 is renewed.

Decision:

- Continue bigger-network scaling with one long D224/L12 no-checkpoint run.
- Do not spend the next run on seed, LR, loop-loss, gate, or objective tables.
- Keep `LOOP_LOSS=all` as the EqR-aligned default.
- Keep checkpoints remote only; Git receives metadata, logs, metrics, and
  visualizations.

## 9. Submission Record

Not a submission run.
