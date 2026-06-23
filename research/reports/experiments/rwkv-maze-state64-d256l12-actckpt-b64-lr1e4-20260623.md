# rwkv-maze-state64-d256l12-actckpt-b64-lr1e4-20260623

## 1. Metainfo

- Plan ID: P-MAZE-009
- Status: failed
- Machine: AIStation GPU1 only
- Start timestamp: 2026-06-23T02:12:51Z
- Local branch: `codex/gpu1-experiment-tracking`
- Source SHA: `e3420cff9581853726c20a13825182a581ec2ccc`

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

Actual:

- `runs/rwkv-maze-state64-d256l12-actckpt-b64-lr1e4-20260623/launch.env`
- `runs/rwkv-maze-state64-d256l12-actckpt-b64-lr1e4-20260623/launch.log`
- `runs/rwkv-maze-state64-d256l12-actckpt-b64-lr1e4-20260623/abort.json`
- `runs/rwkv-maze-state64-d256l12-actckpt-b64-lr1e4-20260623/launch.failed_ninja_20260623T0213Z.log`
- `runs/rwkv-maze-state64-d256l12-actckpt-b64-lr1e4-20260623/launch.failed_ninja_20260623T0223Z.log`

## 7. Results

The first two launches exposed an infrastructure issue: the conda `ninja`
binary core-dumped while building `rwkv7_statepassing_clampw_n64.so`. Commit
`e3420cf` added a wrapper fallback that directly runs the generated nvcc/c++
commands when ninja fails. The head_dim64 kernel then compiled and ran.

The actual b64 training run was stopped before step100 for speed low ROI:

| step | loop1 F1 | loop16 F1 | gain | pred frac loop1->16 | FP loop1->16 | FN loop1->16 |
|---:|---:|---:|---:|---|---|---|
| 1 | 0.3350 | 0.3285 | -0.0065 | 0.3707->0.3375 | 257.6->234.1 | 43.3->49.6 |

After about `15.5m`, only step1 had logged. GPU utilization was active and
memory was about `24.6GB`, so this was compute throughput, not CPU fallback or
idle memory.

## 8. Conclusions

Batch64 state64 is not a practical training point on this GPU1 lease. It
answers a useful engineering question: head_dim64 statepassing can compile and
run, but the throughput is too poor for an 800-step experiment. The mechanism
question was continued with the batch32 viability probe under the same plan.

## 9. Submission Record

None.
