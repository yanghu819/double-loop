# rwkv-maze-clean-scale-d256l12-20260623

## 1. Metainfo

- Plan ID: P-MAZE-008
- Status: in progress
- Machine: AIStation GPU1 only
- Start timestamp: 2026-06-22T23:28:20Z
- Local branch: `codex/gpu1-experiment-tracking`
- Parent/source SHA before launch: `096207200ef91146366358d0e273f3aa8f7db8fa`

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
- Batch/eval: batch16, eval_n256, eval_batch16
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

Exact launch command will be recorded from the generated launch script after the
GitHub-truth planning commit is pushed and archived on GPU1.

## 6. Artifacts

Expected:

- `runs/rwkv-maze-clean-scale-d256l12-20260623/rwkv_maze_probe.json`
- `runs/rwkv-maze-clean-scale-d256l12-20260623/README.md`
- `runs/rwkv-maze-clean-scale-d256l12-20260623/visualizations/index.html`
- `runs/rwkv-maze-clean-scale-d256l12-20260623/visualizations/cases.json`
- `runs/rwkv-maze-clean-scale-d256l12-20260623/abort.json` if early gate fires
- `runs/rwkv-maze-clean-scale-d256l12-20260623/launch.log`
- `runs/rwkv-maze-clean-scale-d256l12-20260623/launch.env`

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

None.
