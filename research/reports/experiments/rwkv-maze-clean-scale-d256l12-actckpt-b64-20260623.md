# rwkv-maze-clean-scale-d256l12-actckpt-b64-20260623

## 1. Metainfo

- Plan ID: P-MAZE-008
- Status: failed
- Machine: AIStation GPU1 only
- Start timestamp: 2026-06-23T00:05:10Z
- Local branch: `codex/gpu1-experiment-tracking`
- Source SHA: `68845e4979efebcb30232a52297239367ebf84df`

## 2. Hypothesis

Pure scaling may still be the right Maze direction, but the current D256/L12
train8/eval16 implementation cannot run without activation checkpointing even
at batch14. The feasible scale axis is therefore activation checkpointing plus a
larger batch, not no-checkpoint fit probing.

If this is the right scaling axis, loop16 should gradually reduce false
positives relative to loop1 while preserving recall. If it only reproduces a
broad mask, pure scaling is not yet enough for Maze.

## 3. Configuration

- Dataset: official `maze-30x30-unique-1k`
- Runner: `scripts/rwkv_maze_probe.py`
- Objective: clean all-loop CE with PATH weight `8`
- FutureSeed: enabled, `future_seed_scale=1`
- Feedback/DAT: disabled
- Boundary/budget decoder: disabled
- Model: D256/L12/H8/head_dim32/channel_mult4
- Loops: train loops 8, eval loops 16
- Batch/eval: batch64, eval_n256, eval_batch16
- Activation checkpointing: enabled
- Steps: 1200, log every 100
- Early gate: at step800, abort if loop16 precision remains below `0.38` and
  loop16 has not dropped at least `10` FP/case relative to loop1.

## 4. Environment

- Remote base: `/huyang2/double-loop`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- Python: `/opt/conda/bin/python`
- GPU: AIStation GPU1, A800 80GB
- GPU2: forbidden
- CPU smoke: forbidden
- Cache/artifacts/runs: project-local under `/huyang2/double-loop`

## 5. Commands

Launched from:

```bash
bash /huyang2/double-loop/artifacts/launch/run_rwkv_maze_clean_scale_d256l12_actckpt_b64_20260623.sh
```

The script archives the GitHub-truth SHA into
`/huyang2/double-loop/artifacts/source/rwkv-maze-clean-scale-d256l12-20260623-68845e4`
and writes logs under
`/huyang2/double-loop/runs/rwkv-maze-clean-scale-d256l12-actckpt-b64-20260623`.

## 6. Artifacts

Expected:

- `runs/rwkv-maze-clean-scale-d256l12-actckpt-b64-20260623/rwkv_maze_probe.json`
- `runs/rwkv-maze-clean-scale-d256l12-actckpt-b64-20260623/README.md`
- `runs/rwkv-maze-clean-scale-d256l12-actckpt-b64-20260623/visualizations/index.html`
- `runs/rwkv-maze-clean-scale-d256l12-actckpt-b64-20260623/visualizations/cases.json`
- `runs/rwkv-maze-clean-scale-d256l12-actckpt-b64-20260623/abort.json` if early gate/OOM/infeasible lease fires
- `runs/rwkv-maze-clean-scale-d256l12-actckpt-b64-20260623/launch.log`
- `runs/rwkv-maze-clean-scale-d256l12-actckpt-b64-20260623/launch.env`

## 7. Results

Activation-checkpoint batch64 fit and reached step100, but became numerically
unstable.

| step | CE | loop1 F1 | loop16 F1 | gain | pred frac loop1->16 | FP loop1->16 | FN loop1->16 |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 1.8253 | 0.3137 | 0.3228 | +0.0091 | 0.6673->0.6253 | 487.5->452.6 | 6.2->9.1 |
| 100 | NaN | 0.0000 | 0.0000 | +0.0000 | 0.0000->0.0000 | 0.0->0.0 | 119.3->119.3 |

Artifacts pulled locally under
`runs/rwkv-maze-clean-scale-d256l12-actckpt-b64-20260623/`.

## 8. Conclusions

This run separates resource feasibility from optimizer stability. Activation
checkpointing makes batch64 trainable in memory and step1 already shows loop16
can cut FP, but `lr=3e-4` is unstable by step100.

Decision: one lower-LR stabilization probe is justified; do not treat this as a
mechanism failure.

## 9. Submission Record

None.
