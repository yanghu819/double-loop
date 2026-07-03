# GDN Transition Mixed Hard D224/L12 Step7000

## 1. Metainfo

- run_name: `gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136`
- plan_id: `P-DIAG-018`
- machine: AIStation `GPU1` only
- local_start_time: `2026-07-03 10:13 CST`
- status: `done`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

P-DIAG-017 showed the cleanest current positive signal: a short mixed `51-64` continuation from the D224/L12 step6000 checkpoint improved holes60 more than the much longer pure `56-64` tail, while preserving holes53.

The mechanism hypothesis is simple and bitter-lesson-compatible: after the model has opened `51-55`, a broader hard-data distribution gives the recurrent state enough continuous support to learn harder global consistency. If this is true, longer mixed `51-64` training from step6100 should improve holes60 / official `56-64` without erasing `51-55`. If it fails, same-shape long training is no longer the main path.

## 3. Configuration

- Resume checkpoint: P-DIAG-017 step6100 checkpoint.
- Data: official EqR Sudoku arrays.
- Backbone: native FutureSeed GDN, D224/L12/H14/D16, `GDN_EXPAND_V=4.0`.
- Loop: loop5, `LOOP_LOSS=all`.
- Continuation distribution: `HOLES_MIN=51`, `HOLES_MAX=64`.
- Target: `FULL_STEPS=7000`, with checkpoint eval at `6300,6600,7000`.
- No scratch, no extra loss, no repair/search/selector, no CPU smoke, no GPU2.

Kill criteria:

- At step6300, stop if holes60 loop5 is clearly below step6100 (`0.1973/0.6603`) and holes53 also drops, because that would mean the short positive signal was not stable.
- Stop immediately on NaN/OOM/low GPU utilization with high memory.

## 4. Environment

- remote work_dir: `/huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step7000-20260703T0443Z-b040136`
- source SHA: `b040136922b614bb30e0dbab4d60c0d438ab9ff4`
- GPU: AIStation `GPU1`, `CUDA_VISIBLE_DEVICES=0`.
- Python: `/opt/conda/bin/python`.
- Repo path: `/huyang2/double-loop`.

## 5. Commands

Executed launch:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
SUDOKU_SIZE=9 PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
RESUME_TRAIN_CHECKPOINT=/huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369/runs/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369/checkpoints/train_state_step006100.pt \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0 \
D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all SCRATCH_MODE=none \
HOLES_MIN=51 HOLES_MAX=64 EVAL_HOLES=60 EVAL_HOLES_LIST=53,60 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=7000 FULL_EVAL_N=512 FULL_ROLLOUT_KS="" \
EVAL_CHECKPOINT_STEPS=6300,6600,7000 EVAL_CHECKPOINT_HOLES_LIST=53,60 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 CASE_BANK_HOLES=53,60 CASE_BANK_N=4 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,3,5 \
RUN_NAME=gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136 ./run.sh full
```

First launch `gdn-transition-mixedhard-d224l12-step7000-20260703T0443Z-b040136` omitted `SUDOKU_SIZE=9` and failed at the official-data guard before training. GPU memory stayed at `0 MiB`; it is archived as a launch failure, not an experiment result.

## 6. Artifacts

- Remote run dir: `/huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step7000-20260703T0443Z-b040136/runs/gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136`
- Local run dir: `runs/gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136`
- Initial failed launch archive: `runs/gdn-transition-mixedhard-d224l12-step7000-20260703T0443Z-b040136/abort.json`
- Metadata tar: `.codex-transfer/gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136-metadata-light.tgz`
- Metadata sha256: `b838ab1fe309b85e38d6463410e1b0867695e634a99aca02956f9bcf90f65dc1`
- Checkpoint evals: `output/checkpoint_eval_step006300.json`, `output/checkpoint_eval_step006600.json`, `output/checkpoint_eval_step007000.json`
- Main visualization: `runs/gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136/visualizations/index.html`
- Case banks: `output/case_bank/official_b46_50/index.html`, `output/case_bank/official_b51_55/index.html`, `output/case_bank/official_b56_64/index.html`

## 7. Results

Training curve:

| step | CE | total | loop1 loss | loop last loss |
|---:|---:|---:|---:|---:|
| 6200 | `0.7169` | `0.7964` | `0.9997` | `0.7169` |
| 6300 | `0.7533` | `0.8314` | `1.0338` | `0.7533` |
| 6400 | `0.6361` | `0.7405` | `0.9915` | `0.6361` |
| 6500 | `0.6102` | `0.7236` | `0.9784` | `0.6102` |
| 6600 | `0.6861` | `0.7682` | `0.9918` | `0.6861` |
| 6700 | `0.6003` | `0.7116` | `0.9790` | `0.6003` |
| 6800 | `0.5912` | `0.6973` | `0.9626` | `0.5912` |
| 6900 | `0.6212` | `0.7340` | `1.0026` | `0.6212` |
| 7000 | `0.5736` | `0.6898` | `0.9704` | `0.5736` |

Checkpoint evals:

| step | holes | loop1 exact/blank | loop3 exact/blank | loop5 exact/blank |
|---:|---|---:|---:|---:|
| 6300 | holes53 | `0.0156 / 0.5344` | `0.0918 / 0.6332` | `0.1738 / 0.6462` |
| 6300 | holes60 | `0.0195 / 0.5460` | `0.1074 / 0.6576` | `0.1895 / 0.6702` |
| 6600 | holes53 | `0.0176 / 0.5345` | `0.1172 / 0.6379` | `0.1816 / 0.6468` |
| 6600 | holes60 | `0.0156 / 0.5439` | `0.1172 / 0.6561` | `0.2031 / 0.6666` |
| 7000 | holes53 | `0.0176 / 0.5367` | `0.1133 / 0.6428` | `0.1914 / 0.6568` |
| 7000 | holes60 | `0.0215 / 0.5432` | `0.1133 / 0.6547` | `0.1953 / 0.6670` |

Final mixed full-board metrics:

| loop | exact | blank_acc |
|---:|---:|---:|
| 1 | `0.0234` | `0.5427` |
| 2 | `0.0293` | `0.6080` |
| 3 | `0.1152` | `0.6345` |
| 4 | `0.1855` | `0.6471` |
| 5 | `0.2031` | `0.6479` |

Official blank-range eval:

| range | exact | blank_acc |
|---|---:|---:|
| `46-50` | `1.0000` | `1.0000` |
| `51-55` | `0.3105` | `0.7322` |
| `56-64` | `0.0859` | `0.5938` |

Case-bank summary:

| range | exact | blank_acc | selected |
|---|---:|---:|---|
| `46-50` | `1.0000` | `1.0000` | solved_by_loop `4`, almost `0`, hard `0` |
| `51-55` | `0.3164` | `0.7389` | solved_by_loop `4`, almost `4`, hard `4` |
| `56-64` | `0.0781` | `0.6033` | solved_by_loop `4`, almost `4`, hard `4` |

Final score recorded by `record_experiment.py`: `0.203125` at `metrics.eval_clean.loop5.label_exact`.

Primary readouts:

- holes53 loop1/3/5 exact and blank accuracy.
- holes60 loop1/3/5 exact and blank accuracy.
- official blank ranges `46-50`, `51-55`, `56-64`.
- loop gain and hard-case wrong-cell trajectories.

## 8. Conclusions

Positive, but not a reason to blindly run the same shape forever.

The mixed `51-64` direction is real: step6600 holes60 loop5 exact/blank reaches `0.2031 / 0.6666`, above the step6100 gate and above the pure `56-64` tail step8000 result `0.1836 / 0.6396`. It also preserves/improves the transition range: holes53 loop5 improves from step6100 `0.1641 / 0.6342` to step7000 `0.1914 / 0.6568`, and official `51-55` reaches `0.3105 / 0.7322`.

The result is non-monotonic on the hardest bucket: holes60 exact peaks at step6600 and eases to `0.1953` at step7000; official `56-64` remains only `0.0859`. This means broader hard data is better than the narrow pure hard tail, but same-shape long training is not enough to claim the hardest range is solved.

Decision: keep mixed hard curriculum as the current clean scaling lesson. Do not run a blind D224/L12 9000/12000 repeat. The next high-ROI experiment should either use a staged/broader hard curriculum that preserves `51-55` while gradually increasing hard mass, or change the generic recurrent state formulation. Still no Sudoku repair/search/selector, no scratch gate sweep, no loss-weight table.

## 9. Submission Record

No submission/tag planned unless the score is unexpectedly strong and the mechanism conclusion remains clean.
