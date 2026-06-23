# rwkv-maze-state64-d256l12-actckpt-b32-lr1e4-20260623

## 1. Metainfo

- Plan ID: P-MAZE-009
- Status: done
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

Actual:

- `runs/rwkv-maze-state64-d256l12-actckpt-b32-lr1e4-20260623/launch.env`
- `runs/rwkv-maze-state64-d256l12-actckpt-b32-lr1e4-20260623/launch.log`
- `runs/rwkv-maze-state64-d256l12-actckpt-b32-lr1e4-20260623/abort.json`

The run was manually stopped after step200 because GPU1 had about `786s`
remaining and waiting for step300 risked platform timeout. The step100/200
readouts already answered the viability question.

## 7. Results

| step | loss | loop1 F1 | loop16 F1 | gain | pred frac loop1->16 | FP loop1->16 | FN loop1->16 |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 1.8267 | 0.3341 | 0.3305 | -0.0035 | 0.3784->0.3455 | 263.6->239.7 | 42.3->48.0 |
| 100 | 0.3741 | 0.4681 | 0.4681 | +0.0000 | 0.4328->0.4328 | 270.2->270.2 | 0.0->0.0 |
| 200 | 0.3701 | 0.4686 | 0.4686 | +0.0000 | 0.4322->0.4322 | 269.7->269.7 | 0.0->0.0 |

Comparison to the previous state32 D256/L12 activation-checkpoint b64 run:

| run | step | loop16 F1 | pred frac loop16 | FP loop16 | FN loop16 | loop gain |
|---|---:|---:|---:|---:|---:|---:|
| state32 b64 | 200 | 0.3144 | 0.7097 | 519.4 | 0.0 | +0.0171 |
| state64 b32 | 200 | 0.4686 | 0.4322 | 269.7 | 0.0 | +0.0000 |

## 8. Conclusions

This is a high-information mixed result.

Positive: increasing recurrent state capacity from 32x32 heads to 64x64 heads
changes the learned operating point dramatically. The model no longer sits in
the extreme state32 broad-mask regime: loop16 F1 rises from `0.3144` to
`0.4686`, pred_path_frac drops from `0.7097` to `0.4322`, and FP/case drops
from `519.4` to `269.7` with zero FN.

Negative: the improvement is not late-loop correction. At step100 and step200,
loop1 and loop16 are identical. So expanded state capacity helps the one-pass
path-weighted solution become EqR-like, but it does not make recurrent loops
repair false positives.

Throughput boundary: head_dim64 is expensive. b64 did not reach step100 in
`~15.5m`; b32 reached step200 in `~29m`. Do not continue head_dim64 as the
default Maze scaling axis.

Next decision: keep the insight that recurrent state capacity matters, but
scale it through a more efficient axis, likely head_dim32 with more heads/width
or a simple learned recurrent decision state. Do not spend the next budget on
head_dim64 lengthening, selector, search, repair, or loss-weight sweeps.

## 9. Submission Record

None.
