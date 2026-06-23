# official-eqr-sudoku-quick5-20260623

## 1. Metainfo

- Plan ID: `P-EQR-010`
- Status: in-progress
- Machine: AIStation `GPU1` only
- Start time UTC: `2026-06-23T09:30:00Z`
- Local branch: `codex/gpu1-experiment-tracking`
- Remote work dir: `/huyang2/double-loop/official_eqr_sudoku_repro_20260623`
- Source SHA for launcher/tracking: pending detached launch SHA; recorded in
  remote `source.sha` after launch.

## 2. Hypothesis

The official EqR Sudoku-Extreme released checkpoint should reproduce the README
quick 5-seed mean/std under the official codebase, data, checkpoint, and
`config/eval/sudoku_lite_noise05.yaml`. This is the baseline gate before any
FutureSeed-vs-EqR claim.

## 3. Configuration

- Official EqR upstream SHA: `aba94e9cde0f273ce644db5261cd6915ba6561f0`
- Dataset: official `sudoku-extreme-1k-aug-1000`
- Checkpoint: official HF `locuslab/EqR-model/sudoku-extreme/eqr.pth`
- Eval config: `config/eval/sudoku_lite_noise05.yaml`
- Eval settings: `halt_max_steps=64`, `different_init=128`,
  `convergence_top_k=4`, `noise_scale=0.5`, `init_std=1.0`,
  `global_batch_size=128`, `max_eval_steps=16`
- Seeds: reuse completed seed `0`; run missing seeds `1,2,3,4`
- Compile: `DISABLE_COMPILE=1`
- GPU: `CUDA_VISIBLE_DEVICES=0`

## 4. Environment

- GPU: `NVIDIA A800-SXM4-80GB`
- Python/Torch env: `/huyang2/double-loop/official_eqr_compare/.venv`
- Attention path: PyTorch SDPA compatibility fallback. The literal
  FlashAttention path is blocked on this host by wheel glibc and nvcc/Torch CUDA
  mismatch, but the same fallback reproduced official Maze exact metrics.

## 5. Commands

```bash
cd /huyang2/double-loop/.worktrees/official-eqr-sudoku-quick5-ca6128f
bash runs/official-eqr-sudoku-quick5-20260623/launch_official_sudoku_quick5.sh
```

## 6. Artifacts

- Remote launcher output:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/artifacts/sudoku_quick5_20260623`
- Remote per-seed logs:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/artifacts/sudoku_quick5_20260623/logs`
- Remote per-seed metrics:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/artifacts/sudoku_quick5_20260623/metrics`
- Local launcher:
  `runs/official-eqr-sudoku-quick5-20260623/launch_official_sudoku_quick5.sh`

## 7. Results

Pending. Seed0 was already reproduced in `official-eqr-sudoku-repro-20260623`.
This run fills seeds 1-4 and will aggregate all 5 seeds.

## 8. Conclusions

Pending.

## 9. Submission Record

Not applicable.
