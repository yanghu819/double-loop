# GDN Unseen-Data Coverage Continuation

## 1. Metainfo

- Plan ID: `P-SCALE-036`
- Status: in progress
- Date: 2026-07-23
- Machine: AIStation GPU1 A800 only
- Parent: clean FutureSeed-GDN exact step30000 train-state checkpoint
- Branch: `codex/gpu1-data-coverage`
- Training source SHA: `4ef23df55f3db517a7da5041f3385b9a959ba031`
- Formal worktree:
  `/huyang2/double-loop/.worktrees/pscale036-formal-4ef23df-20260723`

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

The exact RNG reconstruction gate passed before GPU training:

- checkpoint SHA256:
  `b3811baf739de916b5c5724a1bf1e6926fb9959093a2c793b5dadd1e4fac0f23`;
- index:
  `/huyang2/double-loop/data/sudoku-extreme-full/indices/unseen-after-step030000-seed52-b128-51-64.npy`;
- index SHA256:
  `60d65c91d9921109d50b485f5f746219ab7e68f4da2beee90a556a222546901b`;
- manifest:
  `/huyang2/double-loop/data/sudoku-extreme-full/indices/unseen-after-step030000-seed52-b128-51-64.json`;
- target 51-64 rows: `3,741,961`;
- unique target rows seen before step30000: `2,359,779` (`63.0626%`);
- strict unseen rows: `1,382,182`.

GPU1 CUDA smoke:

- run:
  `gdn-data-coverage-unseen-smoke-s30001-20260723T1-4ef23df`;
- result: one resumed BF16 optimizer step completed through the local
  `triton_recurrent` GDN path, followed by the full mixed, official
  blank-range, and case-bank evaluation; exit code `0`.

Formal control:

- run:
  `gdn-data-coverage-control-s31500-20260723T1030Z-4ef23df`;
- launch PID: `3514`;
- launch log:
  `/huyang2/double-loop/artifacts/launch/p036-control-20260723T1030Z/train.log`;
- state at launch: exact source SHA, `git_dirty=0`, GPU1 only.
- metadata archive:
  `/huyang2/double-loop/artifacts/p036-control-metadata-20260723T1030Z.tgz`;
- metadata archive SHA256:
  `ac529242f23cac8c024a905e7903cbec5f4c49c9984bef6d32efe07352563ea6`.

First formal unseen attempt:

- run:
  `gdn-data-coverage-unseen-s31500-20260723T1204Z-4ef23df`;
- worktree:
  `/huyang2/double-loop/.worktrees/pscale036-unseen-4ef23df-20260723`;
- launch PID: `5193`;
- launch log:
  `/huyang2/double-loop/artifacts/launch/p036-unseen-20260723T1204Z/train.log`;
- state at launch: exact source SHA, `git_dirty=0`, strict unseen index,
  GPU1 only.
- status: aborted by exact process group at step30200 before any checkpoint;
- reason: the all-unseen pool did not preserve the control blank-count
  distribution. The full 51-64 pool has mean `55.9917` blanks and 51-55 rows
  are `43.00%`; all strict-unseen rows have mean `56.2601` blanks and 51-55
  rows are only `32.01%`. Continuing would confound independent coverage with
  a harder curriculum.

Corrected formal unseen pool:

- strict unseen rows remain the source set;
- deterministically select `1,000,000` rows with selection seed `2052`;
- match the full control pool's exact per-blank-count proportions across
  blank counts 51 through 64 using largest-remainder integer allocation;
- record the input/output histograms and output index SHA256 in a schema-v2
  manifest;
- rerun from the original step30000 checkpoint, not from the invalid attempt.

## 7. Results

The matched control completed with exit code `0`:

| Control readout | Value |
|---|---:|
| mixed loop1 exact | `0.0234` |
| mixed loop2 exact | `0.0547` |
| mixed loop3 exact | `0.2988` |
| mixed loop4 exact | `0.4297` |
| mixed loop5 exact | `0.4590` |
| official 46-50 loop5 exact | `1.0000` |
| official 51-55 loop5 exact | `0.6094` |
| official 56-64 loop5 exact | `0.3086` |
| train CE | `0.5161` |
| train time | `5224.2s` |
| peak allocated CUDA memory | `44527.1MB` |

Relative to the clean step30000 parent, control changes mixed exact
`0.4805 -> 0.4590`, official 51-55 `0.6152 -> 0.6094`, and official 56-64
`0.3848 -> 0.3086`. Blind continuation from the original replacement pool
therefore does not extend the clean scaling curve at this endpoint.

The distribution-matched strict unseen arm is pending; no mechanism decision
is valid until its endpoint and official evaluation complete.

## 8. Decision

Success requires unseen-data to beat control by at least `+0.03` on mixed or
official 56-64 loop5 exact while preserving official 51-55 within `0.03`.
If all deltas are below `0.01`, stop this sampling direction rather than sweep
samplers, seeds, or curriculum weights.

## 9. Publication Record

Not a submission artifact. A positive result would support a clean
data-efficiency claim: effective independent coverage matters beyond nominal
token count for recurrent reasoning scale.
