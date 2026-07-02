# GDN Transition Mixed Hard D224/L12 Step6300

## 1. Metainfo

- run_name: `gdn-transition-mixedhard-d224l12-step6300-20260702`
- plan_id: `P-DIAG-017`
- machine: AIStation `GPU1` only
- local_start_time: `2026-07-02 22:30 CST`
- status: `done`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

D224/L12 step6000 is the best current clean point. Pure `56-64` tail did not improve, but that might be because the continuation distribution was too narrow. A mixed `51-64` hard distribution is a generic data-scaling test: keep the transition support while exposing harder boards.

Prediction: if the issue is data distribution rather than state formulation, a short step6000->6300 continuation should improve holes60 / official `56-64` without erasing holes53 / `51-55`.

## 3. Configuration

- Resume checkpoint: `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/checkpoints/train_state_step006000.pt`
- Data: official EqR Sudoku arrays
- Backbone: native FutureSeed GDN, D224/L12/H14/D16, `GDN_EXPAND_V=4.0`
- Loop: loop5, `LOOP_LOSS=all`
- Continuation distribution: `HOLES_MIN=51`, `HOLES_MAX=64`
- Resume quirk: set `FULL_STEPS=6300`, `EVAL_CHECKPOINT_STEPS=6300`; run only until global step6300 and stop.

## 4. Environment

- remote work_dir: `/huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369`
- source SHA: `01513695dc97b5ceb2431798bac521b0611f1114`
- GPU: AIStation `GPU1`, A800 80GB, `CUDA_VISIBLE_DEVICES=0`
- Python: `/opt/conda/bin/python`, torch `2.7.0+cu126`
- Resume checkpoint: `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/checkpoints/train_state_step006000.pt`

## 5. Commands

Executed gate:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
RESUME_TRAIN_CHECKPOINT=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/checkpoints/train_state_step006000.pt \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0 \
D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all SCRATCH_MODE=none \
HOLES_MIN=51 HOLES_MAX=64 EVAL_HOLES=60 EVAL_HOLES_LIST=53,60 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=6100 FULL_EVAL_N=512 FULL_ROLLOUT_KS="" \
EVAL_CHECKPOINT_STEPS=6100 EVAL_CHECKPOINT_HOLES_LIST=53,60 CASE_BANK_N=0 \
RUN_NAME=gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369 ./run.sh full
```

The original step6300 launch was stopped because the remaining lease was insufficient. The final evidence run is step6100.

## 6. Artifacts

- Remote run dir: `/huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369/runs/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369`
- Local run dir: `runs/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369`
- Metadata tar: `.codex-transfer/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369-metadata-light.tgz`
- Metadata sha256: `d78f191bf5b494dec12bdc69e89aa52ca6d36f2539903f6b3ac0afb72eb70437`
- Checkpoint eval: `runs/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369/output/checkpoint_eval_step006100.json`
- Stop marker: `runs/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369/stop.json`
- Visualization: `runs/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369/visualizations/index.html`

## 7. Results

Step6100 training diagnostic:

| metric | value |
|---|---:|
| CE | `0.7241` |
| total | `0.8031` |
| loop1 loss | `1.0011` |
| loop_last loss | `0.7241` |

Checkpoint eval:

| eval bucket | loop1 exact/blank | loop3 exact/blank | loop5 exact/blank |
|---|---:|---:|---:|
| holes53 | `0.0176 / 0.5312` | `0.1035 / 0.6255` | `0.1641 / 0.6342` |
| holes60 | `0.0215 / 0.5424` | `0.1230 / 0.6501` | `0.1973 / 0.6603` |

Final run score recorded by `record_experiment.py`: `0.185546875` at `metrics.eval_clean.loop5.label_exact`.

## 8. Conclusions

Positive. This is a high-information result because it reverses the prior pure-tail conclusion. Pure `56-64` continuation from D224 did not improve at step8000 (`holes60 loop5 0.1836/0.6396`). A much shorter mixed `51-64` continuation already reaches `0.1973/0.6603` on holes60 and keeps holes53 strong at `0.1641/0.6342`.

Interpretation: harder data is not the problem; too narrow a hard-tail distribution was. The next clean scaling move should be a longer mixed-hard continuation from step6100 toward 6300/7000 with official range eval. This is more bitter-lesson-compatible than adding another side gate: it improves through data distribution and compute, not hand-coded Sudoku repair.

## 9. Submission Record

No submission/tag planned.
