# rwkv-maze-clean-scale-d256l12-20260623

## 1. Metainfo

- Plan ID: P-MAZE-008
- Status: aborted
- Machine: AIStation GPU1 only
- Start timestamp: 2026-06-22T23:28:20Z
- Local branch: `codex/gpu1-experiment-tracking`
- Source SHA: `68845e4979efebcb30232a52297239367ebf84df`

## 2. Hypothesis

Official Maze30 may require pure scale rather than another small objective patch.
The 30x30 grid has 900 tokens and path choice is a global interaction problem.
If scale is the missing ingredient, a larger FutureSeed recurrent model should
not just improve broad coverage; later loops should reduce false positives while
preserving recall.

This is a single scaling gate, not a hidden-size table.

## 3. Configuration

- Dataset: official `maze-30x30-unique-1k`
- Runner: `scripts/rwkv_maze_probe.py`
- Objective: clean all-loop CE with PATH weight `8`
- FutureSeed: enabled, `future_seed_scale=1`
- Feedback/DAT: disabled
- Boundary/budget decoder: disabled
- Model: D256/L12/H8/head_dim32/channel_mult4
- Loops: train loops 8, eval loops 16
- Batch/eval: batch16, eval_n256, eval_batch8
- Activation checkpointing: enabled
- Steps: 1200, log every 100
- Early gate: at step800, abort if loop16 precision remains below `0.38` and
  loop16 has not dropped at least `10` FP/case relative to loop1.

## 4. Environment

- Remote base: `/huyang2/double-loop`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- Python: `/opt/conda/bin/python`
- GPU2: forbidden
- CPU smoke: forbidden
- Cache/artifacts/runs: project-local under `/huyang2/double-loop`

## 5. Commands

Launched from:

```bash
bash /huyang2/double-loop/artifacts/launch/run_rwkv_maze_clean_scale_d256l12_20260623.sh
```

## 6. Artifacts

Actual:

- `runs/rwkv-maze-clean-scale-d256l12-20260623/abort.json`
- `runs/rwkv-maze-clean-scale-d256l12-20260623/launch.log`
- `runs/rwkv-maze-clean-scale-d256l12-20260623/launch.env`

## 7. Results

| step | CE | loop1 F1 | loop16 F1 | gain | pred frac loop1->16 | FP loop1->16 | FN loop1->16 |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 1.8267 | 0.3162 | 0.3279 | +0.0118 | 0.6577->0.6113 | 479.4->440.2 | 6.7->9.4 |

The run was intentionally stopped after step1 because activation-checkpoint
batch16 used only about `12.2` GB on an 80GB card. It was a useful sanity signal
for loop pruning, but not a real scale-up run.

## 8. Conclusions

The initial loop-pruning sign was positive: loop16 cut about `39` FP/case at
step1. However, the configuration was too conservative for an 80GB GPU, so it
could not answer the user's scale-up question.

Decision: abort and relaunch the same hypothesis with more aggressive memory
usage.

## 9. Submission Record

None.
