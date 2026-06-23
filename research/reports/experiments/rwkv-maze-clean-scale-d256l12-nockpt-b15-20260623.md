# rwkv-maze-clean-scale-d256l12-nockpt-b15-20260623

## 1. Metainfo

- Plan ID: P-MAZE-008
- Status: failed
- Machine: AIStation GPU1 only
- Start timestamp: 2026-06-22T23:55:30Z
- Local branch: `codex/gpu1-experiment-tracking`
- Source SHA: `68845e4979efebcb30232a52297239367ebf84df`

## 2. Hypothesis

Pure scale may be enough for official Maze30 if broad-mask behavior is mostly an
under-computation artifact. Batch24 and batch16 no-checkpoint OOMed before
step1, so batch15 no-checkpoint is the largest practical fit point to test this
without falling back to slow activation checkpointing.

The core readout is loop behavior: loop16 must reduce false positives relative
to loop1 while preserving recall. A higher F1 from wider coverage alone is not
treated as success.

## 3. Configuration

- Dataset: official `maze-30x30-unique-1k`
- Runner: `scripts/rwkv_maze_probe.py`
- Objective: clean all-loop CE with PATH weight `8`
- FutureSeed: enabled, `future_seed_scale=1`
- Feedback/DAT: disabled
- Boundary/budget decoder: disabled
- Model: D256/L12/H8/head_dim32/channel_mult4
- Loops: train loops 8, eval loops 16
- Batch/eval: batch15, eval_n256, eval_batch16
- Activation checkpointing: disabled
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
bash /huyang2/double-loop/artifacts/launch/run_rwkv_maze_clean_scale_d256l12_nockpt_b15_20260623.sh
```

The script archives the GitHub-truth SHA into
`/huyang2/double-loop/artifacts/source/rwkv-maze-clean-scale-d256l12-20260623-68845e4`
and writes logs under
`/huyang2/double-loop/runs/rwkv-maze-clean-scale-d256l12-nockpt-b15-20260623`.

## 6. Artifacts

Expected:

- `runs/rwkv-maze-clean-scale-d256l12-nockpt-b15-20260623/rwkv_maze_probe.json`
- `runs/rwkv-maze-clean-scale-d256l12-nockpt-b15-20260623/README.md`
- `runs/rwkv-maze-clean-scale-d256l12-nockpt-b15-20260623/visualizations/index.html`
- `runs/rwkv-maze-clean-scale-d256l12-nockpt-b15-20260623/visualizations/cases.json`
- `runs/rwkv-maze-clean-scale-d256l12-nockpt-b15-20260623/abort.json` if early gate fires or OOM
- `runs/rwkv-maze-clean-scale-d256l12-nockpt-b15-20260623/launch.log`
- `runs/rwkv-maze-clean-scale-d256l12-nockpt-b15-20260623/launch.env`

## 7. Results

Failed before step1 with CUDA OOM.

| readout | value |
|---|---:|
| batch | 15 |
| activation checkpoint | disabled |
| failure | `oom_batch15_no_activation_checkpoint` |
| free memory at failure | 59 MiB |
| requested allocation | 54 MiB |
| PyTorch allocated | 78.69 GiB |

Artifacts pulled locally under
`runs/rwkv-maze-clean-scale-d256l12-nockpt-b15-20260623/`.

## 8. Conclusions

Batch15 is close to the 80GB boundary but still fails before step1. Continuing
to search adjacent no-checkpoint batch values is low ROI.

Decision: stop fine-grained no-checkpoint batch fit attempts after batch14.

## 9. Submission Record

None.
