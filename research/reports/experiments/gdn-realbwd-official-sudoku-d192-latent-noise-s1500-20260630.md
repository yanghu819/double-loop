# GDN D192 Official Sudoku FutureSeed Latent Noise Probe 2026-06-30

## 1. Metainfo

- Plan ID: P-GDN-016
- Run name: `gdn-d192-official-sudoku-fs-latentnoise-s1500-20260630T0256Z`
- Machine: AIStation GPU1 only
- Local repo: `/Users/torusmini/Documents/double-loop-gpu1/.codex-transfer/localrepo`
- Remote repo: `/huyang2/double-loop`
- Planned start: 2026-06-30 10:56 CST / 2026-06-30T02:56Z

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
- Single intervention: `NOISE_SCALE=0.03`, existing feature-diff latent noise in `depth_update`
- Checkpoints: eval at steps `800,1200`
- Case bank: `CASE_BANK_N=3`, `CASE_BANK_EVAL_N=256`, `CASE_BANK_LOOP_VALUES=1,2,3,5`

## 4. Environment

- GPU: GPU1 only via `CUDA_VISIBLE_DEVICES=0`
- No CPU smoke
- Cache/env/model/artifacts remain under `/huyang2/double-loop`
- Expected SHA: to be filled after pre-launch commit

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
NOISE_SCALE=0.03 LOOP_FEEDBACK_SCALE=0 SCRATCH_MODE=none EXACT_MARGIN_WEIGHT=0 \
RUN_NAME=gdn-d192-official-sudoku-fs-latentnoise-s1500-20260630T0256Z-<sha> \
./run.sh full
```

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

No submission planned. No tag unless score becomes genuinely strong and mechanism claim is clean.
