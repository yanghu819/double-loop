# GDN Transition Bridge Hard D224/L12 Step9000

## 1. Metainfo

- run_name: `gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0`
- plan_id: `P-DIAG-019`
- machine: AIStation `GPU1` only
- local_start_time: `2026-07-03 14:04 CST`
- status: `done`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

P-DIAG-018 showed that broad mixed `51-64` training is better than a narrow pure `56-64` tail, but the hardest official range remained weak. The working hypothesis was that `56-64` boards need a continuous bridge from the already-learned `51-55` transition. If the model is pushed directly into hardest-only data, it can lose useful support; if it sees a staged bridge, it may preserve transition skill while shifting the boundary upward.

This is still a bitter-lesson experiment: more data distribution, more compute, same generic model. No Sudoku repair, no selector, no handwritten rule.

## 3. Configuration

- Resume checkpoint: P-DIAG-018 `train_state_step007000.pt`.
- Data: official EqR Sudoku arrays.
- Backbone: native FutureSeed GDN, D224/L12/H14/D16, `GDN_EXPAND_V=4.0`.
- Loop: loop5, `LOOP_LOSS=all`.
- Curriculum:
  - historical skipped prefix: `46-50:100,51-55:5900,51-64:1000`
  - active continuation: `51-55:600,51-64:800,56-64:600`
- Target: `FULL_STEPS=9000`, checkpoint eval at `7600,8400,9000`.
- No scratch, no extra loss, no repair/search/selector, no CPU smoke, no GPU2.

## 4. Prediction And Kill Criteria

Prediction:

- Success if official `56-64` loop5 exact moved from P-DIAG-018 `0.0859` to `>=0.12`, while official `51-55` remained `>=0.25`.
- Stronger success if holes60 loop5 exact reached `>=0.23` without blank accuracy dropping.

Kill criteria:

- If step7600 holes53 collapsed far below P-DIAG-018 final `0.1914/0.6568`, the bridge stage was harming transition skill.
- If step8400 holes60 did not beat P-DIAG-018 best `0.2031/0.6666` and CE/blank had no slope, stop instead of burning a same-shape 9000 repeat.
- Stop immediately on NaN, OOM, or GPU utilization anomaly.

## 5. Commands

Executed through:

```bash
bash /huyang2/double-loop/artifacts/launch/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/launch.sh
```

Important environment:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean
SUDOKU_SIZE=9 PYTHON_BIN=/opt/conda/bin/python
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0
D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all SCRATCH_MODE=none
HOLE_STAGES=46-50:100,51-55:5900,51-64:1000,51-55:600,51-64:800,56-64:600
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=9000 FULL_EVAL_N=512
EVAL_CHECKPOINT_STEPS=7600,8400,9000 EVAL_CHECKPOINT_HOLES_LIST=53,60
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64
```

## 6. Artifacts

- Remote run dir: `/huyang2/double-loop/.worktrees/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/runs/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0`
- Local run dir: `runs/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0`
- Metadata tar: `.codex-transfer/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0-metadata-light.tgz`
- Metadata sha256: `7305966db37da1559a2ac6bb6d38df2d6f183d3c26a92dd84facb4504ea6b23e`
- Main visualization: `runs/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/visualizations/index.html`
- Case banks: `output/case_bank/official_b46_50/index.html`, `output/case_bank/official_b51_55/index.html`, `output/case_bank/official_b56_64/index.html`

Checkpoints stayed remote and are excluded from the committed metadata tar.

## 7. Results

Training curve highlights:

| step | stage | CE | total | loop1 loss | loop last loss |
|---:|---|---:|---:|---:|---:|
| 7100 | 51-55 | `0.2914` | `0.4526` | `0.8641` | `0.2914` |
| 7600 | 51-55 | `0.1887` | `0.3673` | `0.8372` | `0.1887` |
| 7700 | 51-64 | `0.5902` | `0.7076` | `0.9970` | `0.5902` |
| 8400 | 51-64 | `0.4094` | `0.5629` | `0.9460` | `0.4094` |
| 8500 | 56-64 | `0.6622` | `0.7768` | `1.0616` | `0.6622` |
| 9000 | 56-64 | `0.2735` | `0.4980` | `1.0432` | `0.2735` |

Checkpoint evals:

| step | holes | loop1 exact/blank | loop3 exact/blank | loop5 exact/blank |
|---:|---|---:|---:|---:|
| 7600 | holes53 | `0.0176 / 0.5294` | `0.1250 / 0.6049` | `0.1953 / 0.6184` |
| 7600 | holes60 | `0.0195 / 0.5411` | `0.1348 / 0.6212` | `0.2012 / 0.6345` |
| 8400 | holes53 | `0.0176 / 0.5384` | `0.1367 / 0.6329` | `0.2031 / 0.6432` |
| 8400 | holes60 | `0.0215 / 0.5479` | `0.1445 / 0.6544` | `0.2246 / 0.6686` |
| 9000 | holes53 | `0.0176 / 0.5316` | `0.1191 / 0.6031` | `0.1836 / 0.6152` |
| 9000 | holes60 | `0.0215 / 0.5420` | `0.1445 / 0.6218` | `0.2090 / 0.6294` |

Final mixed full-board metrics:

| loop | exact | blank_acc |
|---:|---:|---:|
| 1 | `0.0234` | `0.5381` |
| 2 | `0.0332` | `0.5905` |
| 3 | `0.1523` | `0.6114` |
| 4 | `0.1934` | `0.6207` |
| 5 | `0.1973` | `0.6231` |

Official blank-range eval:

| range | exact | blank_acc |
|---|---:|---:|
| `46-50` | `1.0000` | `1.0000` |
| `51-55` | `0.2871` | `0.6942` |
| `56-64` | `0.1250` | `0.5624` |

Case-bank summary:

| range | exact | blank_acc | selected |
|---|---:|---:|---|
| `46-50` | `1.0000` | `1.0000` | solved_by_loop `4`, almost `0`, hard `0` |
| `51-55` | `0.3281` | `0.7176` | solved_by_loop `4`, almost `4`, hard `4` |
| `56-64` | `0.1406` | `0.5732` | solved_by_loop `4`, almost `4`, hard `4` |

Final score recorded by `record_experiment.py`: `0.197265625` at `metrics.eval_clean.loop5.label_exact`.

## 8. Conclusions

The staged bridge curriculum worked. It moved official `56-64` from P-DIAG-018 `0.0859/0.5938` to `0.1250/0.5624` while keeping official `51-55` above the planned guard (`0.2871/0.6942`, guard was `>=0.25`). This is a clean positive scaling result: no new module, no repair, no selector, no Sudoku rule.

The best intermediate checkpoint is step8400, not final step9000. Step8400 holes60 loop5 reaches `0.2246/0.6686`, beating P-DIAG-018 best holes60 `0.2031/0.6666`. The final `56-64` tail improves official hardest exact but reduces mixed checkpoint quality: holes60 loop5 falls to `0.2090/0.6294`, and final mixed score is `0.1973`.

Decision: bridge-hard curriculum is a real data-scaling lever. Do not run long pure `56-64` tails. If continuing this family, use a cyclic or weighted bridge-hard curriculum that keeps `51-55` support while sampling enough `56-64`, and treat step8400-style checkpoints as better candidates than final tail checkpoints. This supports the bitter-lesson line: scale data/compute through the right distribution, not problem-specific postprocessing.

## 9. Submission Record

No tag. Primary score is below `0.50`.
