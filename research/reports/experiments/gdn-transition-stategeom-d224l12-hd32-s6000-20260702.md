# GDN FutureSeed D224/L12 State Geometry Scale

## 1. Metainfo

- Plan ID: `P-DIAG-014`
- Status: stopped by low-ROI boundary
- Local branch: `codex/gpu1-experiment-tracking`
- Planned time: `2026-07-02 19:00 +0800`
- Stop time: `2026-07-02T12:00:58Z`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: `c50b48b5cc8275ae84b12ec105ba412c291a38e2`
- Run:
  `gdn-transition-stategeom-d224l12-hd32-s6000-20260702T1108Z-c50b48b`

## 2. Hypothesis

P-DIAG-012 proved that D224/L12 with native FutureSeed and enough hard-stage
training can turn the `51-55` cliff into board-level exact. P-DIAG-013 then
showed that simply continuing the same state shape into a `56-64` hard tail did
not improve holes60 at step8000.

The next clean scaling question was whether the bottleneck is recurrent state
geometry. The previous D224/L12 used `HEADS=14`, `HEAD_DIM=16`,
`GDN_EXPAND_V=4`, so each layer had state size `14 * 64 * 16 = 14336`.
Keeping token width fixed but switching to `HEADS=7`, `HEAD_DIM=32` changes
that to `7 * 128 * 32 = 28672`. This doubles generic recurrent memory/state
capacity without adding Sudoku rules, repair, selector, or a hand objective.

Prediction:

- If state geometry is the bottleneck, the H7/D32 run should beat the H14/D16
  D224 baseline at the same step count, especially on holes53/holes60 loop5.
- If it only slows training or stays on the same curve, the blocker is not
  simple state-matrix capacity. Then the next direction should be a cleaner
  recurrent state formulation, not another width/state table.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=224`, `LAYERS=12`, `HEADS=7`, `HEAD_DIM=32`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`
- Loop supervision: `LOOP_LOSS=all`
- Curriculum: `HOLE_STAGES=46-50:100,51-55:5900`
- Train target: `FULL_STEPS=6000`, microbatch `FULL_BATCH=32`,
  `GRAD_ACCUM_STEPS=4`, effective batch `128`
- Eval: official test split, `FULL_EVAL_N=512`
- Checkpoint eval: steps `1000,3000,4500,6000`, holes `53,60`
- Official blank-range eval: `46-50,51-55,56-64`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`, no activation checkpoint
- Optimization: `LR=0.0015`, `WEIGHT_DECAY=0.001`, `BLANK_LOSS_WEIGHT=8`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule, no seed/LR/loss/gate table.

## 4. Environment

- GPU: AIStation `GPU1`
- `CUDA_VISIBLE_DEVICES=0`
- Observed memory: about `71.5GB / 80GB`
- Post-stop GPU state: `0 MiB` used
- SSH helper worked after GPU1 restart; no WebShell fallback needed for launch.

## 5. Commands

Run shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=224 LAYERS=12 HEADS=7 HEAD_DIM=32 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 LOOP_LOSS=all \
HOLE_STAGES=46-50:100,51-55:5900 \
FULL_STEPS=6000 FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_EVAL_N=512 \
EVAL_HOLES=53 EVAL_HOLES_LIST=53,60 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=1000,3000,4500,6000 \
EVAL_CHECKPOINT_HOLES_LIST=53,60 \
SAVE_TRAIN_CHECKPOINT_EVERY=500 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 LR=0.0015 WEIGHT_DECAY=0.001 \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=3 CASE_BANK_EVAL_N=192 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=gdn-transition-stategeom-d224l12-hd32-s6000-20260702T1108Z-c50b48b \
./run.sh full
```

Stop action:

- Step1000 checkpoint eval completed.
- Exact PID chain was stopped after writing `early_stop.json`.
- Stop reason: step1000 matched the H14/D16 D224 baseline quality while being
  much slower and more memory-heavy.

## 6. Artifacts

- Remote run dir:
  `/huyang2/double-loop/.worktrees/gdn-transition-stategeom-d224l12-hd32-c50b48b-20260702T1108Z/runs/gdn-transition-stategeom-d224l12-hd32-s6000-20260702T1108Z-c50b48b`
- Remote metadata tar:
  `/huyang2/double-loop/artifacts/transfer/gdn-transition-stategeom-d224l12-hd32-s6000-20260702T1108Z-c50b48b-metadata-light.tgz`
- Local metadata tar:
  `.codex-transfer/gdn-transition-stategeom-d224l12-hd32-s6000-20260702T1108Z-c50b48b-metadata-light.tgz`
- SHA256:
  `7908a9112a4b6289a6bc03b9f9d98881ab1fdf346598ee5e8c0b6e8833d0a0e8`
- Checkpoints remain remote only and are not committed.

## 7. Results

Train:

| Step | Stage | CE | Total | Loop1 | Loop5 |
|---:|---|---:|---:|---:|---:|
| 100 | 46-50 | 1.4033 | 1.4022 | 1.4005 | 1.4033 |
| 200 | 51-55 | 1.1522 | 1.1539 | 1.1607 | 1.1522 |
| 300 | 51-55 | 1.0606 | 1.0633 | 1.0738 | 1.0606 |
| 400 | 51-55 | 1.0203 | 1.0260 | 1.0462 | 1.0203 |
| 500 | 51-55 | 0.9996 | 1.0074 | 1.0354 | 0.9996 |
| 600 | 51-55 | 0.9580 | 0.9697 | 1.0116 | 0.9580 |
| 700 | 51-55 | 0.9579 | 0.9699 | 1.0120 | 0.9579 |
| 800 | 51-55 | 0.9424 | 0.9587 | 1.0144 | 0.9424 |
| 900 | 51-55 | 0.9322 | 0.9489 | 1.0025 | 0.9322 |
| 1000 | 51-55 | 0.9090 | 0.9291 | 0.9899 | 0.9090 |

Step1000 checkpoint eval:

| Eval bucket | Loop | Exact | Blank acc | Clue ok |
|---|---:|---:|---:|---:|
| holes53 | 1 | 0.0039 | 0.4928 | 1.0000 |
| holes53 | 2 | 0.0176 | 0.5207 | 1.0000 |
| holes53 | 3 | 0.0176 | 0.5273 | 1.0000 |
| holes53 | 4 | 0.0176 | 0.5266 | 1.0000 |
| holes53 | 5 | 0.0176 | 0.5263 | 1.0000 |
| holes60 | 1 | 0.0098 | 0.5020 | 1.0000 |
| holes60 | 2 | 0.0215 | 0.5314 | 1.0000 |
| holes60 | 3 | 0.0215 | 0.5375 | 1.0000 |
| holes60 | 4 | 0.0215 | 0.5388 | 1.0000 |
| holes60 | 5 | 0.0215 | 0.5390 | 1.0000 |

Comparison to D224/H14/D16 baseline:

- H14/D16 D224 step1000 holes53 loop5 exact/blank was `0.0176/0.5284`.
- H7/D32 step1000 holes53 loop5 exact/blank is `0.0176/0.5263`.
- H7/D32 uses about `71.5GB`, versus about `44.8GB` observed for H14/D16.
- H7/D32 step1000 elapsed was about `3212s`, so it is too slow for a clean
  6000-step run in one GPU1 lease.

## 8. Conclusions

This is a useful negative scaling boundary.

Doubling recurrent state geometry is trainable and does not collapse: CE falls
from `1.4033` to `0.9090`, and loop1 to loop5 still improves exact/blank on
both holes53 and holes60. But it does not beat the cheaper H14/D16 state shape
at the first fair checkpoint. It spends much more memory and wall time to land
on the same quality curve.

Decision:

- Stop H7/D32 D224/L12 state-geometry scaling; do not run it to 3000/6000.
- Do not turn this into a head_dim/head_count table.
- Keep P-DIAG-012 as the current best clean scaling result.
- Next high-ROI direction should be a more efficient recurrent state
  formulation or a simple learned state update, not brute-force larger
  per-head state geometry.

## 9. Submission Record

Not a submission run. No tag: score is below threshold and the run is a
boundary, not a new best.
