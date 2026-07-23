# GDN Unseen-Data Coverage Continuation

## 1. Metainfo

- Plan ID: `P-SCALE-036`
- Status: approved
- Date: 2026-07-23
- Machine: AIStation GPU1 A800 only
- Parent: clean FutureSeed-GDN exact step30000 train-state checkpoint
- Branch: `codex/gpu1-data-coverage`

## 2. Hypothesis

The clean step30000 run consumed about one dataset-equivalent number of board
draws, but it sampled with replacement. Nominal compute therefore overstates
the number of independent relations actually observed.

If the remaining hard-Sudoku error is still data-limited, directing the next
1500 optimizer steps to rows never drawn in the first 30000 steps should beat
an equal-step continuation from the full pool. If it does not, additional
sampler engineering is low ROI and the next axis must change generic model
capacity or state formulation.

## 3. Configuration

Both arms resume the same model, AdamW state, feature-buffer state, and RNG:

- local Triton-recurrent GDN, D224/L12/H14/D16, expand-v4;
- native FutureSeed scale 1 and fixed unit normalization;
- five loops with equal CE supervision at every loop;
- BF16, microbatch 32, gradient accumulation 4, effective batch 128;
- official full-diversity 9x9 Sudoku;
- hard stage 51-64 blanks;
- no noise, scratch, feedback, selector, rollout, repair, search, or task rule.

The only difference is the training row pool:

- `control`: original full 51-64 candidate pool with replacement;
- `unseen`: a strict subset containing only rows never drawn before step30000.

## 4. Environment

- Remote root: `/huyang2/double-loop`
- Worktree: recorded after launch
- Python: `/opt/conda/bin/python`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- GPU2 and CPU model smoke: forbidden

## 5. Commands

Build the unseen index only after exact RNG validation:

```bash
/opt/conda/bin/python scripts/build_unseen_official_sudoku_index.py \
  --data-dir /huyang2/double-loop/data/sudoku-extreme-full \
  --checkpoint /huyang2/double-loop/models/gdn-full-diversity-d224l12-s6000-20260711T1510Z-eeb38f5/checkpoints/train_state_step030000.pt \
  --output /huyang2/double-loop/data/sudoku-extreme-full/indices/unseen-after-step030000-seed52-b128-51-64.npy \
  --manifest /huyang2/double-loop/data/sudoku-extreme-full/indices/unseen-after-step030000-seed52-b128-51-64.json
```

Matched formal arms:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_gdn_data_coverage.sh control
CUDA_VISIBLE_DEVICES=0 ./scripts/run_gdn_data_coverage.sh unseen
```

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Decision

Success requires unseen-data to beat control by at least `+0.03` on mixed or
official 56-64 loop5 exact while preserving official 51-55 within `0.03`.
If all deltas are below `0.01`, stop this sampling direction rather than sweep
samplers, seeds, or curriculum weights.

## 9. Publication Record

Not a submission artifact. A positive result would support a clean
data-efficiency claim: effective independent coverage matters beyond nominal
token count for recurrent reasoning scale.
