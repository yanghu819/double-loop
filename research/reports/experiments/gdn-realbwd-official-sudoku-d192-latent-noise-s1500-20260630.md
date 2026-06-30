# GDN D192 Official Sudoku FutureSeed Latent Noise Probe 2026-06-30

## 1. Metainfo

- Plan ID: P-GDN-016
- Run name: `gdn-d192-official-sudoku-fs-latentnoise001-s1500-20260630T0315Z`
- Machine: AIStation GPU1 only
- Local repo: `/Users/torusmini/Documents/double-loop-gpu1/.codex-transfer/localrepo`
- Remote repo: `/huyang2/double-loop`
- Planned start: 2026-06-30 11:15 CST / 2026-06-30T03:15Z

## 2. Hypothesis

The official EqR loss is not a complex pruning/repair objective. It is token CE plus halt BCE, while recurrent latent states are trained under noise. Our recent GDN+FutureSeed Sudoku runs were mostly clean and may therefore learn a deterministic operating point that stops improving after early loops.

If EqR-style latent noise is the missing training pressure, a single noisy-state training run should improve loop5 exact or at least make hard cases continue correcting from loop3 to loop5. If it only changes CE/blank accuracy while exact stays on the `0.0293` plateau, latent noise alone is not the bottleneck and more noise-scale sweeps are low ROI.

## 3. Configuration

- Data: official EqR Sudoku arrays under `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000`
- Task: 9x9 Sudoku, official train/test split, full official blank range
- Model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`
- Size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`, `CHANNEL_MULT=4`, `L_CYCLES=2`
- Loop: `MAX_LOOPS=5`, fixed native FutureSeed
- Train: `FULL_STEPS=1500`, `FULL_BATCH=128`, `FULL_EVAL_N=2048`, `FORWARD_DTYPE=bfloat16`
- Loss: normal blank-weighted CE, no exact-margin, no feedback corruption, no scratch objective
- Single intervention: `NOISE_SCALE=0.01`, existing feature-diff latent noise in `depth_update`, matching the official EqR training default magnitude
- Checkpoints: eval at steps `800,1200`
- Case bank: `CASE_BANK_N=3`, `CASE_BANK_EVAL_N=256`, `CASE_BANK_LOOP_VALUES=1,2,3,5`

## 4. Environment

- GPU: GPU1 only via `CUDA_VISIBLE_DEVICES=0`
- No CPU smoke
- Cache/env/model/artifacts remain under `/huyang2/double-loop`
- SHA: `19007fc86f55213b6c5afedda9315c419e1fffc5`
- GPU: `NVIDIA A800-SXM4-80GB`
- Python: `/opt/conda/bin/python`
- Torch: `2.7.0+cu126`
- Prelaunch calibration: a `NOISE_SCALE=0.03` attempt from SHA `0e205c2` was stopped by exact PID after step800 because it was clearly too strong (`CE=1.2037`, clean checkpoint loop5 exact `0.0000`, blank `0.4408`). It wrote `abort.json` remotely and is treated as an early abort, not the formal quality run.

## 5. Commands

```bash
SMOKE_DONE=1 SKIP_SETUP=1 CUDA_VISIBLE_DEVICES=0 \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 HOLE_PATTERN=random HOLES_MIN=45 HOLES_MAX=64 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=1.0 \
D_MODEL=192 LAYERS=10 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
MAX_LOOPS=5 FULL_STEPS=1500 FULL_BATCH=128 FULL_EVAL_N=2048 \
EVAL_CHECKPOINT_STEPS=800,1200 CASE_BANK_N=3 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,2,3,5 \
FORWARD_DTYPE=bfloat16 LR=0.0015 WEIGHT_DECAY=0.001 BLANK_LOSS_WEIGHT=8 \
FUTURE_SEED_SCALE=1 FUTURE_SEED_UPDATE=fixed \
NOISE_SCALE=0.01 LOOP_FEEDBACK_SCALE=0 SCRATCH_MODE=none EXACT_MARGIN_WEIGHT=0 \
RUN_NAME=gdn-d192-official-sudoku-fs-latentnoise001-s1500-20260630T0315Z-<sha> \
./run.sh full
```

## 6. Artifacts

- Remote formal run:
  `/huyang2/double-loop/.worktrees/gdn-latentnoise001-19007fc-20260630/runs/gdn-d192-official-sudoku-fs-latentnoise001-s1500-20260630T0315Z-19007fc`
- Local formal run:
  `runs/gdn-d192-official-sudoku-fs-latentnoise001-s1500-20260630T0315Z-19007fc`
- Local visualization:
  `runs/gdn-d192-official-sudoku-fs-latentnoise001-s1500-20260630T0315Z-19007fc/visualizations/index.html`
- Local output JSON/MD/HTML:
  `runs/gdn-d192-official-sudoku-fs-latentnoise001-s1500-20260630T0315Z-19007fc/output/`
- 0.03 abort archive:
  `runs/gdn-d192-official-sudoku-fs-latentnoise-s1500-20260630T0256Z-0e205c2-abort/abort.json`

## 7. Results

### 0.03 Prelaunch Abort

`NOISE_SCALE=0.03` was too strong for this backbone/objective:

| step | train CE | clean loop5 exact | clean loop5 blank |
|---:|---:|---:|---:|
| 800 | `1.2037` | `0.0000` | `0.4408` |

It was stopped by exact PID after the step800 checkpoint. This is useful calibration, not the formal result.

### 0.01 Formal Run

Train:

| metric | value |
|---|---:|
| train CE | `0.9861` |
| loop1 train loss | `1.2190` |
| loop5 train loss | `0.9861` |
| train sec | `1039.1` |
| CUDA max allocated | `48031 MB` |
| feature buffer count | `8192` |

Checkpoint eval:

| checkpoint | train CE | loop5 exact | loop5 blank |
|---:|---:|---:|---:|
| 800 | `0.9850` | `0.02246` | `0.52463` |
| 1200 | `0.9493` | `0.02148` | `0.52842` |

Final clean eval:

| loop | exact | blank_acc |
|---:|---:|---:|
| 1 | `0.00537` | `0.44747` |
| 2 | `0.02881` | `0.51383` |
| 3 | `0.02930` | `0.53417` |
| 4 | `0.02930` | `0.53503` |
| 5 | `0.02930` | `0.53514` |

Noisy eval at `NOISE_SCALE=0.01`:

| loop | exact | blank_acc |
|---:|---:|---:|
| 5 | `0.02979` | `0.53297` |

Case bank summary:

| split | eval_n | final exact | final blank |
|---|---:|---:|---:|
| official case bank | 256 | `0.03516` | `0.54005` |

Selected case trajectories:

| case | blanks | wrong loop1->2->3->5 | changed loop1->2->3->5 |
|---|---:|---|---|
| solved_by_loop_01_b0089 | 47 | `5 -> 0 -> 0 -> 0` | `0 -> 5 -> 0 -> 0` |
| solved_by_loop_02_b0131 | 49 | `4 -> 0 -> 0 -> 0` | `0 -> 4 -> 0 -> 0` |
| solved_by_loop_03_b0095 | 47 | `4 -> 0 -> 0 -> 0` | `0 -> 4 -> 0 -> 0` |
| hard_failure_01_b0022 | 58 | `29 -> 16 -> 6 -> 6` | `0 -> 31 -> 17 -> 0` |
| hard_failure_02_b0243 | 55 | `17 -> 15 -> 7 -> 6` | `0 -> 19 -> 10 -> 2` |
| hard_failure_03_b0083 | 56 | `23 -> 13 -> 8 -> 7` | `0 -> 21 -> 9 -> 1` |

## 8. Conclusions

Decision: discard as a plateau breaker, keep as a useful training-dynamics lesson.

What worked:
- Official-scale latent noise `0.01` trains healthily. It is completely different from the failed `0.03` run.
- It gives a decent step800 signal: loop5 exact `0.02246`, blank `0.52463`.
- It preserves the known early-loop behavior: loop1 is rough, loop2/3 do most of the work.

What did not work:
- It did not break the D192/L10 plateau. Final clean loop5 exact is exactly `0.029296875`, matching the clean 1500 baseline plateau.
- It did not make late loops keep fixing hard boards. Hard cases mostly improve by loop3, then stall: `6->6`, `7->6`, `8->7` wrong cells from loop3 to loop5.
- Noisy eval gives only `0.02979` exact and lower blank accuracy, so breadth/noise is not a meaningful selector-free win here.

Lesson:
EqR-style latent noise matters as a scale-sensitive training ingredient, but by itself it is not the missing mechanism for full-board exact. The core bottleneck remains converting per-cell confidence into board-level consistency after early loops. Do not sweep `NOISE_SCALE` values. The next bitter-lesson-friendly direction is more real scale/state capacity, or a simpler recurrent state formulation where later loops have more useful state to revise.

## 9. Submission Record

No submission planned. No tag unless score becomes genuinely strong and mechanism claim is clean.
