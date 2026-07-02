# GDN FutureSeed D224/L12 No-Checkpoint Long Scale

## 1. Metainfo

- Plan ID: `P-DIAG-011`
- Status: done
- Local branch: `codex/gpu1-experiment-tracking`
- Planned time: `2026-07-01 19:25 +0800`
- Run start: `2026-07-01 19:25 +0800`
- Run complete: `2026-07-01 21:46 +0800`
- Archived locally: `2026-07-02 09:58 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: `4346594c0a8bbb78a90bad687a064985892b5cd6`
- Run name: `gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594`
- Remote run dir: `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-long-4346594-20260701T1125Z/runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594`
- Local archive: `runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594`
- Parent evidence:
  - P-DIAG-005/P-DIAG-004 showed D192/L10 `GDN_EXPAND_V=4.0`
    has real slope on the 51-55 transition, reaching official `51-55`
    exact `0.0459` after resume scaling.
  - P-DIAG-010 showed D224/L12 no-checkpoint fits in 80GB, stays finite,
    and drops hard-stage CE from `1.4890` at step150 to `1.0738` at step300.
  - P-DIAG-006/P-DIAG-007 NaNs were tied to activation checkpointing, not
    necessarily to depth/width itself.

## 2. Hypothesis

The current bottleneck is not that FutureSeed/GDN cannot learn Sudoku at all;
it is that the model hits a global-consistency cliff once blanks cross roughly
50. The most bitter-lesson-compatible next axis is more effective recurrent
model capacity and longer hard-stage training, using the clean D224/L12
configuration that already passed the short viability gate.

Prediction:

- If capacity is the limiting factor, D224/L12 long resume should beat the
  D192/L10 transition result on holes53 or official `51-55` exact.
- If exact remains flat while blank accuracy improves, then larger same-family
  capacity is not enough; the next change should be a cleaner generic recurrent
  state formulation rather than another width/depth/step table.

This is a single scale test, not an ablation sweep.

## 3. Configuration

- Resume checkpoint:
  `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-e97469d-20260701T1050Z/runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/checkpoints/train_state_step000300.pt`
- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=224`, `LAYERS=12`, `HEADS=14`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`
- Loop supervision: `LOOP_LOSS=all`
- Curriculum: `HOLE_STAGES=46-50:100,51-55:2900`
- Train: total `FULL_STEPS=3000`, microbatch `FULL_BATCH=32`,
  `GRAD_ACCUM_STEPS=4`, effective batch `128`
- Eval: official test split, `FULL_EVAL_N=512`
- Checkpoint eval: steps `1000,2000,3000`, holes53
- Official blank-range eval: `46-50,51-55,56-64`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`, no activation checkpoint
- Optimization: `LR=0.0015`, `WEIGHT_DECAY=0.001`, `BLANK_LOSS_WEIGHT=8`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule, no seed/LR/loss/gate table.

## 4. Environment

- GPU: AIStation `GPU1`, A100/A800 80GB class
- CUDA visibility: `CUDA_VISIBLE_DEVICES=0`
- Python: `/opt/conda/bin/python`
- Torch: `2.7.0+cu126`
- Device: `cuda`
- GDN mode: `triton_recurrent`
- Forward dtype: `bfloat16`
- Activation checkpoint: disabled
- Git dirty at launch: false

## 5. Commands

Remote launch shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
RESUME_TRAIN_CHECKPOINT=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-e97469d-20260701T1050Z/runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/checkpoints/train_state_step000300.pt \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 LOOP_LOSS=all \
HOLE_STAGES=46-50:100,51-55:2900 \
FULL_STEPS=3000 FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_EVAL_N=512 \
HOLES_MIN=46 HOLES_MAX=55 EVAL_HOLES=53 EVAL_HOLES_LIST=50,53,56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=1000,2000,3000 EVAL_CHECKPOINT_HOLES_LIST=53 \
SAVE_TRAIN_CHECKPOINT_EVERY=500 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 LR=0.0015 WEIGHT_DECAY=0.001 \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=3 CASE_BANK_EVAL_N=192 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- Stop exact PID on OOM or NaN.
- Stop if checkpoint1000 holes53 exact remains near zero and both CE and blank
  accuracy slope have clearly flattened.
- Stop if throughput is too slow to finish useful eval within the current
  GPU1 lease.

Success criteria:

- Minimal: no OOM/NaN through step1000 and checkpoint eval written.
- Positive: step3000 holes53 loop5 exact `>= 0.0615` or official `51-55`
  exact `> 0.0459`, exceeding the D192/L10 transition resume baseline.
- Strong: official `51-55` exact `>= 0.08` with visual cases showing fewer
  wrong blank cells through later loops.

## 6. Artifacts

- `runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/config.json`
- `runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/logs/run.log`
- `runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/output/checkpoint_eval_step001000.json`
- `runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/output/checkpoint_eval_step002000.json`
- `runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/output/checkpoint_eval_step003000.json`
- `runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/output/case_bank/`
- `runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/visualizations/index.html`
- Remote train checkpoints were intentionally not committed:
  - `checkpoints/train_state_step000500.pt`
  - `checkpoints/train_state_step001000.pt`
  - `checkpoints/train_state_step001500.pt`
  - `checkpoints/train_state_step002000.pt`
  - `checkpoints/train_state_step002500.pt`
  - `checkpoints/train_state_step003000.pt`

## 7. Results

Training CE on the hard `51-55` stage:

| Step | CE | Total | Loop1 CE | Loop-last CE |
|---:|---:|---:|---:|---:|
| 400 | 1.0327 | 1.0378 | 1.0554 | 1.0327 |
| 500 | 1.0119 | 1.0182 | 1.0402 | 1.0119 |
| 600 | 0.9766 | 0.9868 | 1.0248 | 0.9766 |
| 700 | 0.9571 | 0.9681 | 1.0066 | 0.9571 |
| 800 | 0.9606 | 0.9729 | 1.0180 | 0.9606 |
| 900 | 0.9504 | 0.9631 | 1.0083 | 0.9504 |
| 1000 | 0.9498 | 0.9628 | 1.0066 | 0.9498 |
| 1500 | 0.8944 | 0.9134 | 0.9751 | 0.8944 |
| 2000 | 0.8517 | 0.8800 | 0.9631 | 0.8517 |
| 2500 | 0.8441 | 0.8768 | 0.9712 | 0.8441 |
| 3000 | 0.7943 | 0.8400 | 0.9681 | 0.7943 |

Checkpoint eval on holes53:

| Checkpoint | Loop1 exact / blank | Loop3 exact / blank | Loop5 exact / blank |
|---:|---:|---:|---:|
| 1000 | 0.0156 / 0.4985 | 0.0176 / 0.5273 | 0.0176 / 0.5284 |
| 2000 | 0.0176 / 0.5031 | 0.0195 / 0.5439 | 0.0195 / 0.5449 |
| 3000 | 0.0176 / 0.5112 | 0.0293 / 0.5724 | 0.0352 / 0.5750 |

Final mixed official eval:

| Loop | Exact | Blank acc |
|---:|---:|---:|
| 1 | 0.0215 | 0.5215 |
| 2 | 0.0234 | 0.5675 |
| 3 | 0.0371 | 0.5844 |
| 4 | 0.0352 | 0.5882 |
| 5 | 0.0410 | 0.5897 |

Final official blank-range eval:

| Blank range | Exact | Valid | Solved | Blank acc |
|---|---:|---:|---:|---:|
| 46-50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 51-55 | 0.0195 | 0.0195 | 0.0195 | 0.6441 |
| 56-64 | 0.0137 | 0.0137 | 0.0137 | 0.5266 |

Case-bank summary:

- `46-50`: exact `0.9948`, blank `0.9997`, mostly solved.
- `51-55`: exact `0.0260`, blank `0.6522`, includes solved-by-loop,
  almost-solved, and hard failures.
- `56-64`: exact `0.0000`, blank `0.5231`, almost-solved and hard failures.

## 8. Conclusions

This run is a mixed scale result:

- Positive: D224/L12 no-checkpoint remains stable through 3000 total steps.
- Positive: holes53 exact and blank accuracy keep improving from checkpoint1000
  to checkpoint3000, and depth loop now adds useful full-board refinement
  (`loop1 -> loop5` mixed exact `0.0215 -> 0.0410`).
- Negative: the run misses the planned success gate. Step3000 holes53 loop5
  exact `0.0352` is below the D192/L10 resume checkpoint `0.0615`, and
  official `51-55` exact `0.0195` is below the D192/L10 resume final `0.0459`.

Decision:

- Do not claim D224/L12 is better than D192/L10 at 3000.
- Because checkpoint exact still has slope (`0.0176 -> 0.0195 -> 0.0352`)
  and loop refinement is stronger than earlier runs, one continuation to 6000
  is justified by the user's long-run scaling request.
- If D224/L12 still fails to beat the D192/L10 51-55 gate after 6000 total
  steps, same-family bigger-network scaling should be deprioritized in favor
  of a cleaner recurrent state formulation.

## 9. Submission Record

Not a submission run.
