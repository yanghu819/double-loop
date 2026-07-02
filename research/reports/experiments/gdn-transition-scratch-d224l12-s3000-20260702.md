# GDN Transition Scratch D224/L12 S3000

## 1. Metainfo

- run_name: `gdn-transition-scratch-d224l12-s3000-20260702T1340Z`
- plan_id: `P-DIAG-016`
- machine: AIStation `GPU1` only
- local_start_time: `2026-07-02 21:40 CST`
- status: `stopped`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

Raw D256 width, raw H7/D32 state geometry, and same-shape `56-64` hard-tail all failed as next scaling axes after D224/L12 opened `51-55`. The likely bottleneck is not just more channels; it is that later loops lack an independent generic work memory and tend to recompute the same state.

A gated scratch state is a minimal bitter-lesson-compatible modeling change: it gives each loop a learned, task-agnostic working state without Sudoku repair, search, selector, or hand rules. If this is the missing state/update capacity, scratch should become active and loop3->5 should keep improving hard boards earlier than the fixed D224 baseline.

## 3. Configuration

- Data: official EqR Sudoku arrays under `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000`
- Backbone: standalone native FutureSeed GDN, Triton recurrent kernel
- Model: `D_MODEL=224`, `LAYERS=12`, `HEADS=14`, `HEAD_DIM=16`, `CHANNEL_MULT=4`, `GDN_EXPAND_V=4.0`
- Loop: `MAX_LOOPS=5`, `L_CYCLES=2`, `LOOP_LOSS=all`
- Scratch: `SCRATCH_MODE=gated`, `SCRATCH_SCALE=1.0`, no scratch noise, no scratch Gaussian regularizer
- Batch: microbatch `32`, `GRAD_ACCUM_STEPS=4`, effective batch `128`
- Curriculum: `HOLE_STAGES=46-50:100,51-55:2900`
- Eval: checkpoint eval at `1000,2000,3000` on holes `53,60`; final official ranges `46-50,51-55,56-64`; case bank enabled
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair, no Sudoku-specific rule

## 4. Environment

- remote work_dir: `/huyang2/double-loop/.worktrees/gdn-transition-scratch-d224l12-s3000-20260702T134526Z-35756dd`
- source SHA: `35756ddf2adba5eb5a35ecb1390cf6026e7cd354`
- GPU: AIStation `GPU1`, A800 80GB, `CUDA_VISIBLE_DEVICES=0`
- Python: `/opt/conda/bin/python`, torch `2.7.0+cu126`
- Cache/artifacts/runs: all under `/huyang2/double-loop`
- Note: first launch used `/huyang2/double-loop/.venv/bin/python`, which had no torch. It failed before training and was superseded by the `/opt/conda/bin/python` run.

## 5. Commands

Executed command shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
SUDOKU_SIZE=9 HOLE_PATTERN=random \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0 \
D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all \
SCRATCH_MODE=gated SCRATCH_SCALE=1.0 SCRATCH_NOISE_SCALE=0 SCRATCH_GAUSS_WEIGHT=0 \
HOLE_STAGES=46-50:100,51-55:2900 EVAL_HOLES=53 EVAL_HOLES_LIST=53,60 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=3000 FULL_EVAL_N=512 \
EVAL_CHECKPOINT_STEPS=300,600,1000,2000,3000 EVAL_CHECKPOINT_HOLES_LIST=53,60 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 CASE_BANK_N=6 CASE_BANK_EVAL_N=512 \
FULL_ROLLOUT_KS="" \
RUN_NAME=gdn-transition-scratch-d224l12-earlyckpt-20260702T135930Z-35756dd ./run.sh full
```

Kill criteria:

- If the early step300/600 gates are clearly below the fixed D224 curve, stop without waiting for step1000.
- If step1000 holes53 loop5 exact/blank is not competitive with fixed D224 step1000 `0.0176/0.5284`, or holes60 loop5 is not competitive with `0.0215/0.5390`, stop.
- If `scratch_delta` is effectively zero at step1000, stop: the mechanism is not being used.
- If memory or speed is materially worse than fixed D224 with no metric signal, stop.

## 6. Artifacts

- Remote run dir: `/huyang2/double-loop/.worktrees/gdn-transition-scratch-d224l12-s3000-20260702T134526Z-35756dd/runs/gdn-transition-scratch-d224l12-earlyckpt-20260702T135930Z-35756dd`
- Local run dir: `runs/gdn-transition-scratch-d224l12-earlyckpt-20260702T135930Z-35756dd`
- Metadata tar: `.codex-transfer/gdn-transition-scratch-d224l12-earlyckpt-20260702T135930Z-35756dd-metadata-light.tgz`
- Metadata sha256: `955dbb167affce41126636ded84a417d03760303eb474e2a0a9ccb7244b253f9`
- Checkpoint eval: `runs/gdn-transition-scratch-d224l12-earlyckpt-20260702T135930Z-35756dd/output/checkpoint_eval_step000300.json`
- Abort: `runs/gdn-transition-scratch-d224l12-earlyckpt-20260702T135930Z-35756dd/abort.json`
- Visualization: `runs/gdn-transition-scratch-d224l12-earlyckpt-20260702T135930Z-35756dd/visualizations/index.html`

## 7. Results

Stopped at step300 by ROI gate.

Training diagnostics at step300:

| metric | value |
|---|---:|
| CE | `1.2726` |
| loop1 loss | `1.2823` |
| loop_last loss | `1.2726` |
| scratch_gate | `0.1251` |
| scratch_decay | `0.8305` |
| scratch_delta | `0.4044` |
| scratch_residual | `7.0730` |

Checkpoint eval:

| eval bucket | loop1 exact/blank | loop3 exact/blank | loop5 exact/blank |
|---|---:|---:|---:|
| holes53 | `0.0000 / 0.3974` | `0.0000 / 0.4007` | `0.0000 / 0.4013` |
| holes60 | `0.0000 / 0.3975` | `0.0000 / 0.4039` | `0.0000 / 0.4043` |

Reference fixed D224 step300 holes53 loop5 was `0.0020 / 0.4704`.

## 8. Conclusions

Discard this mechanism. The scratch state was active, so the negative result is not a wiring failure. It added a learned per-cell working memory (`scratch_delta=0.4044`) but slowed the early transition badly: holes53 blank accuracy is about `-0.069` below fixed D224 at the same step, exact is zero, and loop1->5 gains are tiny.

Decision: do not sweep scratch gate bias, decay bias, scratch noise, Gaussian projections, or seeds. This is another small side-memory gate boundary. The next useful direction should either be a more integrated recurrent state formulation, or a data/curriculum scaling axis that preserves the D224 hard-stage slope.

## 9. Submission Record

No submission/tag planned unless score is meaningfully strong and mechanism evidence is clean.
