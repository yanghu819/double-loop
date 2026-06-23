# official-eqr-maze-repro-20260623

## 1. Metainfo

- Plan ID: `P-EQR-008`
- Status: done
- Machine: AIStation `GPU1` only
- Start time UTC: `2026-06-23T04:12:47Z`
- Local branch: `codex/gpu1-experiment-tracking`
- Remote work dir: `/huyang2/double-loop/official_eqr_repro_20260623`

## 2. Hypothesis

Before comparing FutureSeed to EqR, the official EqR Maze-Unique released
checkpoint must reproduce the paper's exact-accuracy numbers under the official
codebase, official data, official checkpoint, and official evaluation config.
If this gate fails, all downstream FutureSeed-vs-EqR claims are ungrounded.

## 3. Configuration

- Official EqR upstream SHA: `aba94e9cde0f273ce644db5261cd6915ba6561f0`
- Official repo path: `/huyang2/double-loop/official_eqr_repro_20260623/eqr-official`
- Dataset: official `maze-30x30-unique-1k`
- Checkpoint: official HF `locuslab/EqR-model/maze-unique/eqr.pth`
- Checkpoint SHA256: `9cb8ab682519f290c3ec157b5a16829e51afff27773651fbd01a260b8748b138`
- Eval entrypoint: `scripts/eval.sh` using `config/eval/depth_breadth.yaml`
- Critical Maze eval setting: `noise_scale=0.01`. The generic
  `depth_breadth.yaml` default is `0.5`, but that setting is Sudoku-oriented
  and makes the released Maze checkpoint fail exact accuracy.
- Eval points:
  - D16/B1: `halt_max_steps=16 different_init=1 noise_scale=0.01`
  - D64/B1: `halt_max_steps=64 noise_scale=0.01`
  - D64/B128: `halt_max_steps=64 different_init=128 convergence_top_k=1 global_batch_size=16 noise_scale=0.01`

## 4. Environment

- GPU: `NVIDIA A800-SXM4-80GB`
- Python/Torch env: `/huyang2/double-loop/official_eqr_compare/.venv`
- Torch: `2.7.0+cu126`
- PyTorch CXX11 ABI: `True`
- AdamAtan2 backend: present and importable
- FlashAttention: official CUDA path is not available in this container. The
  repo tries `flash_attn_interface` first and then `flash_attn.flash_attn_func`;
  this environment only has system `flash-attn==1.0.4`, which lacks
  `flash_attn_func`. A matching modern wheel
  `flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl`
  was downloaded locally and uploaded, but it requires `GLIBC_2.32` while the
  host has glibc `2.31`. A source build is blocked by the host CUDA compiler
  being `nvcc 11.7` while the Torch wheel is `cu126`.
- Attention fallback: PyTorch SDPA fallback was applied only to unblock
  evaluation on this host. After correcting Maze `noise_scale` to `0.01`, SDPA
  reproduces the paper Maze exact numbers within expected eval variation.

## 5. Commands

```bash
# GPU1 start/probe was done through Kimi WebBridge + AIStation helper.

BASE=/huyang2/double-loop/official_eqr_repro_20260623
git clone --shared /huyang2/double-loop/official_eqr_compare/eqr-clean "$BASE/eqr-official"
cd "$BASE/eqr-official"
git checkout --detach aba94e9cde0f273ce644db5261cd6915ba6561f0
git reset --hard aba94e9cde0f273ce644db5261cd6915ba6561f0
git clean -fdx
ln -sfn /huyang2/double-loop/official_eqr_compare/eqr-clean/data/maze-30x30-unique-1k \
  data/maze-30x30-unique-1k

source /huyang2/double-loop/official_eqr_compare/.venv/bin/activate
python -m pip install --no-deps \
  "$BASE/artifacts/wheelhouse/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  bash scripts/eval.sh "$BASE/downloaded_checkpoints/maze-unique/eqr.pth" \
  noise_scale=0.01

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  bash scripts/eval.sh "$BASE/downloaded_checkpoints/maze-unique/eqr.pth" \
  halt_max_steps=64 noise_scale=0.01

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  bash scripts/eval.sh "$BASE/downloaded_checkpoints/maze-unique/eqr.pth" \
  halt_max_steps=64 different_init=128 convergence_top_k=1 global_batch_size=16 noise_scale=0.01

# Actual successful D16 command also used noise_scale=0.01:
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled DISABLE_COMPILE=1 \
  python evaluate.py eval_yaml=config/eval/depth_breadth.yaml \
  checkpoint="$BASE/downloaded_checkpoints/maze-unique/eqr.pth" \
  global_batch_size=128 noise_scale=0.01 force_rerun=true
```

## 6. Artifacts

- Remote logs under `/huyang2/double-loop/official_eqr_repro_20260623/logs`
- Remote eval outputs under
  `/huyang2/double-loop/official_eqr_repro_20260623/downloaded_checkpoints/eval_preds`
- Local archive under `runs/official-eqr-maze-repro-20260623`
- Launcher: `runs/official-eqr-maze-repro-20260623/launch_official_repro.sh`

## 7. Results

Official paper target for Maze-Unique Table 4:

| Setting | Paper exact | Reproduced exact | Reproduced token acc | Notes |
|---|---:|---:|---:|---|
| D16/B1 | 0.822 | 0.827 | 0.984574 | `noise_scale=0.01`, SDPA fallback |
| D64/B1 | 0.889 | 0.892 | 0.991523 | `noise_scale=0.01`, SDPA fallback |
| D64/B128 | 0.930 | 0.944444 | 0.990981 | `convergence_top_k/exact_accuracy`, `noise_scale=0.01`, SDPA fallback |

Diagnostic that explains the earlier failure:

| Setting | Exact | Token acc | Interpretation |
|---|---:|---:|---|
| D16/B1, `noise_scale=0.5` | 0.000 | 0.863877 | Wrong Maze eval noise for the released checkpoint. Token accuracy is dominated by non-path cells and is not enough. |
| D64/B1, `noise_scale=0.5` | 0.000 | 0.861108 | Same failure. |
| D64/B128, `noise_scale=0.5` | 0.007937 | 0.861101 | Same failure; breadth cannot rescue the wrong noise setting. |

Small-batch saved-prediction audit under the failing `noise_scale=0.5` setting:
labels use PATH token `5`, but most PATH labels were predicted as ordinary
open-cell token `2` (`1791` of `1865` path cells in the 16-sample audit). This
shows why token accuracy looked acceptable while exact accuracy was zero.

## 8. Conclusions

- The official EqR Maze released checkpoint is reproducible on GPU1 once Maze
  eval uses `noise_scale=0.01`, matching the checkpoint training config.
- Reproduced exact accuracy meets or exceeds the paper Maze-Unique Table 4
  targets: D16/B1 `0.827` vs `0.822`, D64/B1 `0.892` vs `0.889`,
  and D64/B128 convergence top-1 `0.944444` vs `0.930`.
- The README's generic `depth_breadth.yaml` default `noise_scale=0.5` is not a
  valid Maze released-checkpoint reproduction setting; it is enough to destroy
  exact accuracy while leaving misleadingly high token accuracy.
- For future paper comparisons, EqR baseline must use official code/data/model
  plus `noise_scale=0.01`. FutureSeed experiments should be compared against
  this baseline, not against our older path-weight proxy runs.
- FlashAttention remains an environment blocker for literal official-kernel
  reproduction on this host, but SDPA fallback is now numerically adequate for
  the Maze exact reproduction gate.

## 9. Submission Record

Not applicable.
