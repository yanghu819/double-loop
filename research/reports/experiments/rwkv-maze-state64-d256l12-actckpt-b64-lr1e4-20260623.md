# rwkv-maze-state64-d256l12-actckpt-b64-lr1e4-20260623

## 1. Metainfo

- Plan ID: P-MAZE-009
- Status: in-progress
- Machine: AIStation GPU1 only
- Start timestamp: 2026-06-23T02:12:51Z
- Local branch: `codex/gpu1-experiment-tracking`
- Source SHA: recorded in `launch.env` at launch

## 2. Hypothesis

The previous D256/L12 clean scale run showed a real but small loop-pruning
signal: loop16 cut about `44` false-positive path cells per case without
creating false negatives, but it stayed in a very broad-mask regime.

This experiment tests whether the bottleneck is not token hidden width but
RWKV recurrent state capacity. With `d_model=256`, switching from
`heads=8, head_dim=32` to `heads=4, head_dim=64` keeps the token width fixed
while doubling each layer's recurrent matrix capacity from `8*32*32=8192` to
`4*64*64=16384`.

If this is the right bitter-lesson scaling axis, longer clean training should
let later loops carry richer path-separation state and push loop16 masks
narrower than the state32 run while preserving recall. If it repeats the same
broad-mask plateau, the Maze bottleneck is more likely the objective or
decision-state dynamics, not raw RWKV state size.

## 3. Configuration

- Dataset: official `maze-30x30-unique-1k`
- Runner: `scripts/rwkv_maze_probe.py`
- Objective: clean all-loop CE with PATH weight `8`
- FutureSeed: enabled, `future_seed_scale=1`
- Feedback/DAT: disabled
- Boundary/budget decoder: disabled
- Model: D256/L12/H4/head_dim64/channel_mult4
- Loops: train loops 8, eval loops 16
- Batch/eval: batch64, eval_n256, eval_batch16
- Activation checkpointing: enabled
- LR: `1e-4`
- Steps: 800, log every 100
- Early gate: keep only if loop16 improves beyond the state32 broad-mask
  plateau; otherwise stop and archive as a negative state-capacity result.

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
bash /huyang2/double-loop/artifacts/launch/run_rwkv_maze_state64_d256l12_actckpt_b64_lr1e4_20260623.sh
```

The script archives the GitHub-truth SHA into
`/huyang2/double-loop/artifacts/source/rwkv-maze-state64-d256l12-20260623-<sha>`
and writes logs under
`/huyang2/double-loop/runs/rwkv-maze-state64-d256l12-actckpt-b64-lr1e4-20260623`.

## 6. Artifacts

Planned:

- `runs/rwkv-maze-state64-d256l12-actckpt-b64-lr1e4-20260623/launch.env`
- `runs/rwkv-maze-state64-d256l12-actckpt-b64-lr1e4-20260623/launch.log`
- `runs/rwkv-maze-state64-d256l12-actckpt-b64-lr1e4-20260623/rwkv_maze_probe.json`
- `runs/rwkv-maze-state64-d256l12-actckpt-b64-lr1e4-20260623/abort.json` if killed
- hard-case visual HTML if the runner reaches or triggers visual output

## 7. Results

Pending.

Primary readouts:

- loop1 vs loop16 path F1
- loop1 vs loop16 precision/recall/pred_path_frac
- loop1 vs loop16 FP/FN hard-case change
- step100/200/300 CE and broad-mask behavior
- wall time and GPU memory versus state32 scale run

## 8. Conclusions

Pending.

Decision rule:

- Continue state-capacity scaling only if loop16 FP drop clearly exceeds the
  state32 run's `~44` FP/case reduction and pred_path_frac moves below `0.65`
  without a meaningful FN increase.
- Stop this axis if state64 repeats `F1<=0.32` and pred_path_frac `>0.65` by
  step200/300, or if runtime/memory cost makes the same signal low ROI.

## 9. Submission Record

None.
