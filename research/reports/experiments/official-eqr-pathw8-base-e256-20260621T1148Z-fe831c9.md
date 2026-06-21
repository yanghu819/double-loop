# official-eqr-pathw8-base-e256-20260621T1148Z-fe831c9

## 1. Metainfo

- Plan ID: P-MAZE-001
- Status: done
- Machine: GPU1 A100
- Start time UTC: 2026-06-21T11:48:00Z
- Source commit: fe831c9
- Remote work dir: /huyang2/double-loop

## 2. Hypothesis

Official EqR Maze currently fails path recovery because plain token CE is
dominated by non-PATH cells. A generic class-balanced path-token loss should make
the official proxy meaningful before comparing FutureSeed.

## 3. Configuration

- Official EqR SHA: aba94e9cde0f273ce644db5261cd6915ba6561f0
- Condition: clean official EqR baseline, no FutureSeed
- Loss patch: ACTLossHead with `path_token_weight=8.0`, `path_token_id=5`
- Epochs: 256
- Expected train steps: 1792
- Global batch: 128
- Eval interval: 250
- Checkpoint interval: 500
- GPU binding: `CUDA_VISIBLE_DEVICES=0`

## 4. Environment

- AIStation row: GPU1 only
- Python env: /huyang2/double-loop/official_eqr_compare/.venv
- Data: /huyang2/double-loop/official_eqr_compare/eqr-clean/data/maze-30x30-unique-1k

## 5. Commands

```bash
PATH_TOKEN_WEIGHT=8.0 PATH_TOKEN_ID=5 \
EPOCHS=256 TRAIN_EPOCHS_PER_ITER=256 GLOBAL_BATCH_SIZE=128 \
EVAL_INTERVAL_STEPS=250 CHECKPOINT_INTERVAL_STEPS=500 \
HEAVY_METRICS_LOG_INTERVAL=100 STEPS_HIST_LOG_INTERVAL_STEPS=100 \
RUN_NAME=official-eqr-pathw8-base-e256-20260621T1148Z-fe831c9 \
bash /huyang2/double-loop/scripts/official_eqr_compare/run_official_eqr_compare.sh train-base
```

## 6. Artifacts

- Remote log: /huyang2/double-loop/official_eqr_compare/logs/official-eqr-pathw8-base-e256-20260621T1148Z-fe831c9.log
- Remote checkpoint: /huyang2/double-loop/official_eqr_compare/outputs/base/outputs/zui8n95t/2026-6-21/11-41-40/checkpoints/step_1792_zui8n95t.pth
- Remote visualization: /huyang2/double-loop/official_eqr_compare/visuals/official-eqr-pathw8-base-e256-final-20260621T1148Z
- Local archive: runs/official-eqr-pathw8-maze-compare-20260621/base
- Combined score: runs/official-eqr-pathw8-maze-compare-20260621/score.json

## 7. Results

Final checkpoint step1792, 256 official test cases, no selector/search/repair.

| loop | path F1 | precision | recall | pred PATH frac | true PATH frac | FP/case | FN/case |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4664 | 0.3050 | 0.9986 | 0.4340 | 0.1326 | 271.5 | 0.16 |
| 4 | 0.4684 | 0.3066 | 1.0000 | 0.4324 | 0.1326 | 269.8 | 0.00 |
| 8 | 0.4684 | 0.3065 | 1.0000 | 0.4324 | 0.1326 | 269.9 | 0.00 |
| 16 | 0.4682 | 0.3064 | 0.9998 | 0.4325 | 0.1326 | 269.9 | 0.03 |

Training eval reached token accuracy `0.6993` by step1750 with exact accuracy
`0.0`. The path-weighted objective opened PATH emission, but the operating point
is a high-recall broad mask.

## 8. Conclusions

Keep this result as the official EqR Maze baseline for the path-aware objective.
It passes the viability gate for a matched FutureSeed comparison because loop16
path F1 is above `0.10` and pred PATH frac is below the all-PATH collapse guard.

Do not overclaim the task is solved. Loop gain is only `+0.0018` F1 from loop1 to
loop16, and the model leaves about 270 false-positive PATH cells per case.

## 9. Submission Record

Not applicable.
