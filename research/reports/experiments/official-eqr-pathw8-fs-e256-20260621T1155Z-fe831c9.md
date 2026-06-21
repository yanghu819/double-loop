# official-eqr-pathw8-fs-e256-20260621T1155Z-fe831c9

## 1. Metainfo

- Plan ID: P-MAZE-002
- Status: done
- Machine: GPU1 A100
- Start time UTC: 2026-06-21T11:55:00Z
- Source commit: fe831c9
- Remote work dir: /huyang2/double-loop

## 2. Hypothesis

Under the same official EqR code path and the same path-aware supervised
objective, FutureSeed should provide cheaper future context. If this mechanism is
real in the official EqR setting, it should either improve final path F1 or reach
the baseline operating point earlier.

## 3. Configuration

- Official EqR SHA: aba94e9cde0f273ce644db5261cd6915ba6561f0
- Condition: official EqR patched with FutureSeed only
- Loss patch: ACTLossHead with `path_token_weight=8.0`, `path_token_id=5`
- Epochs: 256
- Expected train steps: 1792
- Global batch: 128
- Eval interval: 250
- Checkpoint interval: 500
- GPU binding: `CUDA_VISIBLE_DEVICES=0`
- Matched baseline: `official-eqr-pathw8-base-e256-20260621T1148Z-fe831c9`

## 4. Environment

- AIStation row: GPU1 only
- Python env: /huyang2/double-loop/official_eqr_compare/.venv
- Data: /huyang2/double-loop/official_eqr_compare/eqr-futureseed/data/maze-30x30-unique-1k

## 5. Commands

```bash
PATH_TOKEN_WEIGHT=8.0 PATH_TOKEN_ID=5 \
EPOCHS=256 TRAIN_EPOCHS_PER_ITER=256 GLOBAL_BATCH_SIZE=128 \
EVAL_INTERVAL_STEPS=250 CHECKPOINT_INTERVAL_STEPS=500 \
HEAVY_METRICS_LOG_INTERVAL=100 STEPS_HIST_LOG_INTERVAL_STEPS=100 \
RUN_NAME=official-eqr-pathw8-fs-e256-20260621T1155Z-fe831c9 \
bash /huyang2/double-loop/scripts/official_eqr_compare/run_official_eqr_compare.sh train-futureseed
```

## 6. Artifacts

- Remote log: /huyang2/double-loop/official_eqr_compare/logs/official-eqr-pathw8-fs-e256-20260621T1155Z-fe831c9.log
- Remote checkpoint: /huyang2/double-loop/official_eqr_compare/outputs/futureseed/outputs/kr9vj9vj/2026-6-21/11-52-24/checkpoints/step_1792_kr9vj9vj.pth
- Remote visualization: /huyang2/double-loop/official_eqr_compare/visuals/official-eqr-pathw8-fs-e256-final-20260621T1155Z
- Local archive: runs/official-eqr-pathw8-maze-compare-20260621/futureseed
- Combined score: runs/official-eqr-pathw8-maze-compare-20260621/score.json

## 7. Results

Final checkpoint step1792, 256 official test cases, no selector/search/repair.

| loop | path F1 | precision | recall | pred PATH frac | true PATH frac | FP/case | FN/case |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4657 | 0.3044 | 0.9991 | 0.4351 | 0.1326 | 272.4 | 0.11 |
| 4 | 0.4684 | 0.3065 | 0.9999 | 0.4324 | 0.1326 | 269.8 | 0.02 |
| 8 | 0.4683 | 0.3065 | 0.9998 | 0.4324 | 0.1326 | 269.9 | 0.02 |
| 16 | 0.4684 | 0.3065 | 0.9999 | 0.4324 | 0.1326 | 269.9 | 0.02 |

Matched loop16 delta versus clean EqR is `+0.0001` path F1. FutureSeed loop
gain from loop1 to loop16 is `+0.0026`, but this is still an operating-point
stabilization, not meaningful late-loop correction.

## 8. Conclusions

Discard as a positive FutureSeed-over-EqR claim. Under the official EqR mixer,
FutureSeed is neutral once the objective is path-aware.

This is still useful evidence for the paper: EqR is a strong noncausal baseline
and a boundary result. FutureSeed's main claim should be tested where the
backbone lacks cheap future context, such as causal/RWKV recurrent Maze and
Sudoku backbones.

## 9. Submission Record

Not applicable.
