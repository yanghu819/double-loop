# gdn-transition-d256l12-width-s6000-20260702

## 1. Metainfo

- Plan ID: `P-DIAG-015`
- Status: stopped
- Machine: AIStation `GPU1` only
- Started: 2026-07-02T12:14:12Z
- Source SHA at planning time: `ae31c9f54ffb0abaecf48e2102942431f57d36b3`
- Formal run name: `gdn-transition-d256l12-width-noconv-s6000-20260702T1245Z-ae31c9f`

## 2. Hypothesis

D224/L12/H14/D16 long training opened the 51-55 blank transition, while H7/D32 state-geometry scaling tied D224 quality at much worse throughput. The next clean scaling question is whether widening the whole generic token/channel/state path to D256/L12/H16/D16 improves the 51-55/56-64 frontier, or whether the current FutureSeed+GDN loop has moved from capacity-limited into state-update-limited.

## 3. Configuration

- Task: official EqR Sudoku arrays, 9x9, random blanks.
- Backbone: native FutureSeed GDN, real Triton recurrent backward.
- Model: `D_MODEL=256`, `LAYERS=12`, `HEADS=16`, `HEAD_DIM=16`, `GDN_EXPAND_V=4.0`, `CHANNEL_MULT=4`, `MAX_LOOPS=5`.
- Training: `GDN_USE_SHORT_CONV=0`, `LOOP_LOSS=all`, `FULL_BATCH=32`, `GRAD_ACCUM_STEPS=4` if it fits; if OOM, fall back once to `FULL_BATCH=24`, `GRAD_ACCUM_STEPS=6`.
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
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0 \
FORWARD_DTYPE=bfloat16 D_MODEL=256 LAYERS=12 HEADS=16 HEAD_DIM=16 CHANNEL_MULT=4 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all \
HOLE_STAGES=46-50:100,51-55:5900 EVAL_HOLES=53 EVAL_HOLES_LIST=53,60 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=6000 FULL_EVAL_N=2048 FULL_ROLLOUT_KS= \
EVAL_CHECKPOINT_STEPS=1000,3000,4500,6000 EVAL_CHECKPOINT_HOLES_LIST=53,60 \
SAVE_TRAIN_CHECKPOINT_EVERY=1000 TRAIN_CHECKPOINT_DIR=/huyang2/double-loop/models/<run_name>/checkpoints \
CASE_BANK_HOLES=53,60 CASE_BANK_N=8 CASE_BANK_EVAL_N=512 CASE_BANK_LOOP_VALUES=1,2,3,4,5 \
RUN_NAME=gdn-transition-d256l12-width-noconv-s6000-20260702T1236Z-<sha> ./run.sh full
```

Launch note:

- First attempt `gdn-transition-d256l12-width-s6000-20260702T1214Z-ae31c9f` failed before training because the default `GDN_USE_SHORT_CONV=1` requires `fla`, which is intentionally not installed in this environment. It used 3 MiB GPU memory and produced no training result. The formal run uses the same clean GDN path as prior D224 runs: `GDN_USE_SHORT_CONV=0`.

## 6. Artifacts

- Remote run dir: `/huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z/runs/gdn-transition-d256l12-width-noconv-s6000-20260702T1245Z-ae31c9f`
- Remote checkpoint: `/huyang2/double-loop/models/gdn-transition-d256l12-width-noconv-s6000-20260702T1245Z-ae31c9f/checkpoints/train_state_step001000.pt`
- Remote transfer tar: `/huyang2/double-loop/artifacts/transfer/gdn-transition-d256l12-width-noconv-s6000-20260702T1245Z-ae31c9f-metadata-light.tgz`
- Local transfer tar: `.codex-transfer/gdn-transition-d256l12-width-noconv-s6000-20260702T1245Z-ae31c9f-metadata-light.tgz`
- Tar sha256: `597a93ece28005d0ca14a47a4c674525eaf0a32ca7c418b4731feea8e282aabb`
- Local archive: `runs/gdn-transition-d256l12-width-noconv-s6000-20260702T1245Z-ae31c9f/`

## 7. Results

Training stayed finite and fit GPU1 at about `51.8GB`.

| step | stage | CE | loop1 CE | loop5 CE |
|---:|---|---:|---:|---:|
| 100 | 46-50 | 1.4736 | 1.4716 | 1.4736 |
| 300 | 51-55 | 1.0832 | 1.0917 | 1.0832 |
| 500 | 51-55 | 1.0092 | 1.0290 | 1.0092 |
| 650 | 51-55 | 0.9473 | 0.9799 | 0.9473 |
| 1000 | 51-55 | 0.9539 | 0.9931 | 0.9539 |

Checkpoint eval at step1000:

| bucket | loop1 exact | loop1 blank | loop5 exact | loop5 blank | loop gain exact | loop gain blank |
|---|---:|---:|---:|---:|---:|---:|
| holes53 | 0.0156 | 0.4940 | 0.0176 | 0.5150 | +0.0020 | +0.0210 |
| holes60 | 0.0195 | 0.5055 | 0.0215 | 0.5308 | +0.0020 | +0.0253 |

Reference comparisons:

- D224/H14/D16 step1000 holes53 loop5 exact/blank: `0.0176/0.5284`.
- D224/H7/D32 state-geometry step1000 holes60 loop5 exact/blank: `0.0215/0.5390`.
- D256/H16/D16 step1000 holes53 loop5 exact/blank: `0.0176/0.5150`.
- D256/H16/D16 step1000 holes60 loop5 exact/blank: `0.0215/0.5308`.

## 8. Conclusions

Kill criteria:

- Stop immediately on OOM/NaN and archive `abort.json`.
- If `FULL_BATCH=32` OOMs, relaunch once with `FULL_BATCH=24` and `GRAD_ACCUM_STEPS=6`; no batch table.
- At step1000, continue only if holes53 loop5 exact/blank is meaningfully better than D224/H14/D16 step1000 `0.0176/0.5284`, or if CE/loop gain clearly indicates better slope at acceptable speed.
- If step1000 ties D224 while slower or much higher memory, stop as a width-scaling boundary.

Decision value:

- Positive result supports a bitter-lesson scaling story: native FutureSeed+loop improves with larger generic backbone capacity.
- Negative result says the bottleneck is not raw width or per-head state size; next work should change the generic recurrent state/update formulation.

Decision:

- Stopped at step1000 by ROI rule.
- D256/L12 is not a good next scaling axis. It is trainable and fits, but it ties exact and loses blank accuracy against cheaper D224 at the same checkpoint. It also does not beat the H7/D32 state-geometry run on holes60 blank accuracy.
- This is a useful negative boundary, not a FutureSeed failure. D224/L12 long training remains the current best clean scaling result. The next worthwhile move is either a more efficient/generic recurrent state-update formulation, or a data/curriculum scale move that shows slope against D224, not a D256/D320 width table.

## 9. Submission Record

No external submission.
