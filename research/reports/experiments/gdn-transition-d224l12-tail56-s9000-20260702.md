# GDN FutureSeed D224/L12 Tail 56-64 Continuation To 9000

## 1. Metainfo

- Plan ID: `P-DIAG-013`
- Status: stopped by ROI/lease kill rule
- Local branch: `codex/gpu1-experiment-tracking`
- Planned time: `2026-07-02 12:45 +0800`
- Final stop time: `2026-07-02T10:42:12Z`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: `4a0330a57f62c5fb2a6ef0be5e15ef65bfe57def`
- Parent run:
  `gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e`
- Parent checkpoint:
  `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/checkpoints/train_state_step006000.pt`
- Resume run:
  `gdn-transition-d224l12-tail56-s9000-resume7000-20260702T093823Z-4a0330a`
- Resume checkpoint:
  `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-tail56-4a0330a-20260702T0441Z/runs/gdn-transition-d224l12-tail56-s9000-20260702T0441Z-4a0330a/checkpoints/train_state_step007000.pt`

## 2. Hypothesis

P-DIAG-012 proved that the D224/L12 native-FutureSeed GDN recipe is no longer
stuck at the 51-55 blank cliff once we give it enough hard-stage tokens:
official `51-55` loop5 exact reached `0.2832`, and holes53 checkpoint exact
rose monotonically through step6000.

The next question was whether the same clean scaling recipe can move the next
cliff, `56-64`, or whether that bucket needs a larger model/state or a
different recurrent state update.

Prediction:

- Positive: continuing from step6000 with a 56-64 tail should raise holes60 and
  official `56-64` exact, while not destroying the newly opened `51-55`.
- Negative: if holes60 does not improve after a hard-tail checkpoint, this
  shape is probably not converting extra 56-64 exposure into board-level exact.
  Then the next move is bigger effective recurrent state or cleaner state
  dynamics, not more same-shape tail training.

This was a single continuation run. It was not a seed/LR/loss/gate sweep and it
did not add Sudoku-specific rules.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=224`, `LAYERS=12`, `HEADS=14`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`
- Loop supervision: `LOOP_LOSS=all`
- Curriculum: `HOLE_STAGES=46-50:100,51-55:5900,56-64:3000`
- Train: target total `FULL_STEPS=9000`, microbatch `FULL_BATCH=32`,
  `GRAD_ACCUM_STEPS=4`, effective batch `128`
- Eval: official test split, `FULL_EVAL_N=512`
- Checkpoint eval: holes `53,60`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`, no activation checkpoint
- Optimization: `LR=0.0015`, `WEIGHT_DECAY=0.001`, `BLANK_LOSS_WEIGHT=8`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule, no seed/LR/loss/gate table.

## 4. Environment

- GPU: AIStation `GPU1`, `CUDA_VISIBLE_DEVICES=0`
- Torch: `2.7.0+cu126`
- Device path: CUDA, bf16 forward
- Git dirty: `0`
- SSH helper was unavailable because the AIStation SSH password was rejected,
  so launch/monitoring used AIStation WebShell through Kimi WebBridge.

## 5. Commands

Run shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 LOOP_LOSS=all \
HOLE_STAGES=46-50:100,51-55:5900,56-64:3000 \
FULL_STEPS=9000 FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_EVAL_N=512 \
EVAL_HOLES=60 EVAL_HOLES_LIST=53,60 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=8000,9000 EVAL_CHECKPOINT_HOLES_LIST=53,60 \
SAVE_TRAIN_CHECKPOINT_EVERY=500 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 LR=0.0015 WEIGHT_DECAY=0.001 \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
RUN_NAME=gdn-transition-d224l12-tail56-s9000-resume7000-20260702T093823Z-4a0330a \
./run.sh full
```

Early stop:

- Step8000 primary checkpoint eval was complete.
- GPU1 lease had only a few minutes left, making step9000 unlikely.
- Holes60 loop5 exact/blank at step8000 did not improve over the step7000
  checkpoint recorded before the lease interruption.
- Exact PID chain `180 -> 178 -> 147 -> 142` was terminated. After TERM,
  `nvidia-smi` reported `0 MiB` used on GPU1.

## 6. Artifacts

- Remote run dir:
  `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-tail56-resume7000-4a0330a-20260702T093823Z/runs/gdn-transition-d224l12-tail56-s9000-resume7000-20260702T093823Z-4a0330a`
- Local metadata:
  `runs/gdn-transition-d224l12-tail56-s9000-resume7000-20260702T093823Z-4a0330a/`
- Local files archived: `README.md`, `config.json`, `metadata.json`,
  `score.json`, `early_stop.json`, and a compact `logs/run_tail.txt`.
- Checkpoints remain remote only and are not committed.

## 7. Results

Previous step7000 readout from the interrupted first leg:

- holes60 loop5 exact: `0.1875`
- holes60 loop5 blank accuracy: `0.6624`

Step8000 checkpoint eval from the resumed leg:

| Eval bucket | Loop | Exact | Blank acc | Clue ok |
|---|---:|---:|---:|---:|
| holes53 | 1 | 0.0176 | 0.5358 | 1.0000 |
| holes53 | 2 | 0.0254 | 0.5960 | 1.0000 |
| holes53 | 3 | 0.1094 | 0.6186 | 1.0000 |
| holes53 | 4 | 0.1660 | 0.6279 | 1.0000 |
| holes53 | 5 | 0.1797 | 0.6299 | 1.0000 |
| holes60 | 1 | 0.0195 | 0.5467 | 1.0000 |
| holes60 | 2 | 0.0254 | 0.6052 | 1.0000 |
| holes60 | 3 | 0.0996 | 0.6294 | 1.0000 |
| holes60 | 4 | 0.1738 | 0.6379 | 0.9980 |
| holes60 | 5 | 0.1836 | 0.6396 | 0.9980 |

Loop gain at step8000:

- holes53 exact `+0.1621`, blank `+0.0941`
- holes60 exact `+0.1641`, blank `+0.0929`

Train tail:

- step7100 CE `0.6853`, loop1 `1.0118`, loop_last `0.6853`
- step7600 CE `0.5882`, loop1 `1.0318`, loop_last `0.5882`
- step7900 CE `0.4560`, loop1 `0.9890`, loop_last `0.4560`
- step8000 CE `0.5631`, loop1 `1.0249`, loop_last `0.5631`
- step8100 CE `0.3741`, loop1 `0.9943`, loop_last `0.3741`

## 8. Conclusions

The run is not a failure of FutureSeed+loop. It confirms loop computation is
still meaningful on hard holes: loop1 to loop5 gives about `+0.164` exact on
holes60.

But it is a negative boundary for same-shape 56-64 tail training. Step8000
holes60 loop5 exact/blank (`0.1836/0.6396`) did not improve over the step7000
readout (`0.1875/0.6624`). The model can keep lowering CE on hard-stage samples,
but that did not translate into better full-board exact on the target hard
bucket.

Decision:

- Do not run a same D224/L12/expand_v4 9000/12000-step continuation just because
  CE has local slope.
- Keep the strongest current clean result as P-DIAG-012: D224/L12 to step6000
  opened `51-55` and gave mixed loop5 exact `0.2070`.
- Next high-ROI scaling axis should change effective recurrent capacity or the
  generic state formulation, for example wider state with better throughput or
  a simple learned recurrent state update. It should not add Sudoku repair,
  search, selector, hand rules, or loss-weight tables.

## 9. Submission Record

Not a submission run. No tag: score is below the `>=0.50` tag threshold and the
mechanism result is a boundary, not a new best.
