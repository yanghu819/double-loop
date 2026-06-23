# rwkv-maze-state64-d256l12-actckpt-b32-lr1e4-20260623

## 1. Metainfo

- Plan ID: P-MAZE-009
- Status: in-progress
- Machine: AIStation GPU1 only
- Start timestamp: 2026-06-23T02:43:00Z
- Local branch: `codex/gpu1-experiment-tracking`
- Source SHA: `e3420cff9581853726c20a13825182a581ec2ccc`

## 2. Hypothesis

The batch64 state64 run answered an infrastructure and speed question: the
head_dim64 CUDA statepassing kernel can compile and run, but the throughput is
too low for an 800-step run in the current GPU1 window. That does not yet answer
whether expanded RWKV recurrent state has any Maze mechanism value.

This fallback keeps the exact same clean FutureSeed+loop state-capacity axis
and only drops batch to 32 to get a first real readout. It is a viability probe,
not a batch sweep. If batch32 still fails to reach step100/200 with useful
metrics, head_dim64 state scaling should be stopped as low ROI for this proxy.

## 3. Configuration

- Dataset: official `maze-30x30-unique-1k`
- Runner: `scripts/rwkv_maze_probe.py`
- Objective: clean all-loop CE with PATH weight `8`
- FutureSeed: enabled, `future_seed_scale=1`
- Feedback/DAT: disabled
- Boundary/budget decoder: disabled
- Model: D256/L12/H4/head_dim64/channel_mult4
- Loops: train loops 8, eval loops 16
- Batch/eval: batch32, eval_n256, eval_batch16
- Activation checkpointing: enabled
- LR: `1e-4`
- Steps: 300, log every 100

## 4. Environment

- Remote base: `/huyang2/double-loop`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- Python: `/opt/conda/bin/python`
- GPU: AIStation GPU1, A800 80GB
- GPU2: forbidden
- CPU smoke: forbidden
- Cache/artifacts/runs: project-local under `/huyang2/double-loop`

## 5. Commands

Planned launch:

```bash
bash /huyang2/double-loop/artifacts/launch/run_rwkv_maze_state64_d256l12_actckpt_b32_lr1e4_20260623.sh
```

The script archives the GitHub-truth SHA into
`/huyang2/double-loop/artifacts/source/rwkv-maze-state64-d256l12-b32-20260623-e3420cf`
and writes logs under
`/huyang2/double-loop/runs/rwkv-maze-state64-d256l12-actckpt-b32-lr1e4-20260623`.

## 6. Artifacts

Planned:

- `runs/rwkv-maze-state64-d256l12-actckpt-b32-lr1e4-20260623/launch.env`
- `runs/rwkv-maze-state64-d256l12-actckpt-b32-lr1e4-20260623/launch.log`
- `runs/rwkv-maze-state64-d256l12-actckpt-b32-lr1e4-20260623/rwkv_maze_probe.json`
- `runs/rwkv-maze-state64-d256l12-actckpt-b32-lr1e4-20260623/abort.json` if killed

## 7. Results

Pending.

Primary readouts:

- whether step100 arrives within a practical GPU window
- loop1 vs loop16 path F1
- loop1 vs loop16 precision/recall/pred_path_frac
- loop1 vs loop16 FP/FN
- comparison against state32 b64 step100/200 broad-mask plateau

## 8. Conclusions

Pending.

Decision rule:

- Continue head_dim64 state scaling only if batch32 reaches step100/200 and
  loop16 shows stronger FP/pred_frac reduction than state32 without recall
  collapse.
- Stop head_dim64 state scaling if throughput remains poor or if the readout
  repeats broad-mask behavior.

## 9. Submission Record

None.
