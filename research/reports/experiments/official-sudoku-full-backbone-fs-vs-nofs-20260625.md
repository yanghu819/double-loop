# official-sudoku-full-backbone-fs-vs-nofs-20260625

## 1. Metainfo

- Plan ID: `P-SUDOKU-001`
- Status: in progress
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-06-25 10:49:59 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`

## 2. Hypothesis

The previous native FutureSeed EqR gate only tested a 2-layer RWKV mixer
transplant inside official EqR. It should not be used to judge the original
standalone FutureSeed-RWKV-loop method.

Mechanism hypothesis: the complete recurrent backbone needs depth, looped
latent refinement, and official data training together. If the original method
is real rather than a random-hole artifact, it should begin to learn official
Sudoku-Extreme when trained directly on the official arrays. FutureSeed should
improve sample-efficiency or loop refinement relative to the same RWKV-loop
backbone with `future_seed_scale=0`.

Prediction: FutureSeed will show higher official-test blank accuracy and/or
earlier nonzero exact than no-FS under the same budget. If both arms remain
near-zero, the next question becomes clean scaling on official Sudoku, not more
2-layer EqR mixer patching.

## 3. Configuration

- Dataset: official EqR `sudoku-extreme-1k-aug-1000`
- Data mode: map official token `1` to this runner's blank id, and `2..10` to
  digit ids `0..8`
- Model: standalone `FutureSeedLoopSudoku`
- Size: 9x9, 3x3 boxes
- Architecture: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- Train: `FULL_STEPS=1500`, `FULL_BATCH=128`, official train split
- Eval: official test split fixed subset, `FULL_EVAL_N=2048`
- Arms:
  - no-FS: `FUTURE_SEED_SCALE=0`
  - FS: `FUTURE_SEED_SCALE=1`
- Forbidden: right-to-left scan, solver, repair, selector, oracle rollout,
  Sudoku-specific postprocessing, seed sweep

## 4. Environment

Pending GPU1 probe. Expected:

- CUDA device: GPU1 only via `CUDA_VISIBLE_DEVICES=0`
- Python: existing remote environment under `/huyang2/double-loop`
- RWKV kernel: CUDA statepassing
- Caches/artifacts/runs/models remain under `/huyang2/double-loop`

## 5. Commands

Static checks:

```bash
python -m py_compile experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py
bash -n run.sh
```

Main command shape, run once with `FUTURE_SEED_SCALE=0` and once with
`FUTURE_SEED_SCALE=1`:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/eqr-official/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 D_MODEL=192 LAYERS=10 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 FULL_STEPS=1500 FULL_BATCH=128 FULL_EVAL_N=2048 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 RWKV_KERNEL=statepassing FORWARD_DTYPE=bfloat16 \
FUTURE_SEED_SCALE=<0-or-1> RUN_NAME=<run> ./run.sh full
```

Kill criteria:

- If the first 100 steps take more than 20 minutes, stop by exact PID and
  relaunch a smaller viability probe with `FULL_BATCH=80`.
- If both arms are still high-loss with official eval blank accuracy near random
  after 1500 steps, do not sweep seeds; classify the next axis as official
  Sudoku scaling.
- If no-FS does not learn at all but FS does, continue the full-backbone
  direction and scale one axis only.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not applicable.
