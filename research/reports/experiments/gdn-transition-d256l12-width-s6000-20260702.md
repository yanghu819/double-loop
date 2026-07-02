# gdn-transition-d256l12-width-s6000-20260702

## 1. Metainfo

- Plan ID: `P-DIAG-015`
- Status: in-progress
- Machine: AIStation `GPU1` only
- Started: 2026-07-02T12:14:12Z
- Source SHA at planning time: `1491926d3eba0553d4c93f46adee9542cc3f4a21`
- Run name: `gdn-transition-d256l12-width-s6000-20260702T1214Z-1491926`

## 2. Hypothesis

D224/L12/H14/D16 long training opened the 51-55 blank transition, while H7/D32 state-geometry scaling tied D224 quality at much worse throughput. The next clean scaling question is whether widening the whole generic token/channel/state path to D256/L12/H16/D16 improves the 51-55/56-64 frontier, or whether the current FutureSeed+GDN loop has moved from capacity-limited into state-update-limited.

## 3. Configuration

- Task: official EqR Sudoku arrays, 9x9, random blanks.
- Backbone: native FutureSeed GDN, real Triton recurrent backward.
- Model: `D_MODEL=256`, `LAYERS=12`, `HEADS=16`, `HEAD_DIM=16`, `GDN_EXPAND_V=4.0`, `CHANNEL_MULT=4`, `MAX_LOOPS=5`.
- Training: `LOOP_LOSS=all`, `FULL_BATCH=32`, `GRAD_ACCUM_STEPS=4` if it fits; if OOM, fall back once to `FULL_BATCH=24`, `GRAD_ACCUM_STEPS=6`.
- Curriculum: `HOLE_STAGES=46-50:100,51-55:5900`.
- Evaluation: checkpoint eval at `1000,3000,4500,6000` on holes `53,60`; final official blank ranges `46-50,51-55,56-64`; case bank holes `53,60`.
- Forbidden: GPU2, CPU smoke, selector, search, repair, Sudoku-specific rule, seed/LR/loss/head table.

## 4. Environment

- Remote work dir: `/huyang2/double-loop`.
- Repo policy: remote detached checkout from GitHub truth SHA; artifacts/checkpoints remain under `/huyang2/double-loop`.
- GPU check before launch: GPU1 `NVIDIA A800-SXM4-80GB`, 0 MiB used.

## 5. Commands

Planned command shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
UPDATE_LEADERBOARD=0 \
SUDOKU_SIZE=9 HOLE_PATTERN=random \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/data/eqr/sudoku-extreme-1k-aug-1000 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_EXPAND_V=4.0 \
FORWARD_DTYPE=bfloat16 D_MODEL=256 LAYERS=12 HEADS=16 HEAD_DIM=16 CHANNEL_MULT=4 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all \
HOLE_STAGES=46-50:100,51-55:5900 EVAL_HOLES=53 EVAL_HOLES_LIST=53,60 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=6000 FULL_EVAL_N=2048 FULL_ROLLOUT_KS= \
EVAL_CHECKPOINT_STEPS=1000,3000,4500,6000 EVAL_CHECKPOINT_HOLES_LIST=53,60 \
SAVE_TRAIN_CHECKPOINT_EVERY=1000 TRAIN_CHECKPOINT_DIR=/huyang2/double-loop/models/<run_name>/checkpoints \
CASE_BANK_HOLES=53,60 CASE_BANK_N=8 CASE_BANK_EVAL_N=512 CASE_BANK_LOOP_VALUES=1,2,3,4,5 \
RUN_NAME=gdn-transition-d256l12-width-s6000-20260702T1214Z-1491926 ./run.sh full
```

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Kill criteria:

- Stop immediately on OOM/NaN and archive `abort.json`.
- If `FULL_BATCH=32` OOMs, relaunch once with `FULL_BATCH=24` and `GRAD_ACCUM_STEPS=6`; no batch table.
- At step1000, continue only if holes53 loop5 exact/blank is meaningfully better than D224/H14/D16 step1000 `0.0176/0.5284`, or if CE/loop gain clearly indicates better slope at acceptable speed.
- If step1000 ties D224 while slower or much higher memory, stop as a width-scaling boundary.

Decision value:

- Positive result supports a bitter-lesson scaling story: native FutureSeed+loop improves with larger generic backbone capacity.
- Negative result says the bottleneck is not raw width or per-head state size; next work should change the generic recurrent state/update formulation.

## 9. Submission Record

No external submission.
