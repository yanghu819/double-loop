# official-eqr-sudoku-repro-20260623

## 1. Metainfo

- Plan ID: `P-EQR-009`
- Status: done
- Machine: AIStation `GPU1` only
- Start time UTC: `2026-06-23T06:32:16Z`
- Local branch: `codex/gpu1-experiment-tracking`
- Remote work dir: `/huyang2/double-loop/official_eqr_sudoku_repro_20260623`

## 2. Hypothesis

Before using EqR as the paper baseline for FutureSeed, the official EqR
Sudoku-Extreme released checkpoint must reproduce the official Sudoku quick
gate under the official codebase, official data archive, official checkpoint,
and official evaluation config.

This is a baseline gate, not a FutureSeed experiment. If it fails, the next
step is to debug official reproduction, not to run new FutureSeed variants.

## 3. Configuration

- Official EqR upstream SHA: `aba94e9cde0f273ce644db5261cd6915ba6561f0`
- Official repo path:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/eqr-official`
- Dataset: official `sudoku-extreme-1k-aug-1000` extracted from
  `eqr-data-full.tgz`
- Data archive SHA256:
  `1875b5e14fd6ca8b5bacbe2c0fd09efc661faeb7a1d44cff3d530b09525bb293`
- Checkpoint: official HF `locuslab/EqR-model/sudoku-extreme/eqr.pth`
- Checkpoint SHA256:
  `258b1f0a90003a2db0de0c8df7831683ffce00de295ed4bba15dc8e599920d26`
- Eval config: `config/eval/sudoku_lite_noise05.yaml`
- Key official settings: `halt_max_steps=64`, `different_init=128`,
  `convergence_top_k=4`, `noise_scale=0.5`, `init_std=1.0`,
  `global_batch_size=128`, `max_eval_steps=16`
- Seed: `0`

Official README quick target over 5 seeds:

| Metric | README target |
|---|---:|
| `convergence_top_k/cumulative_exact_acc_top1` | `99.19 +/- 0.12` |
| `convergence_top_k/exact_accuracy` | `98.60 +/- 0.04` |
| `majority_vote/exact_accuracy` | `98.67 +/- 0.05` |
| `different_init/any_correct` | `99.20 +/- 0.10` |

The full paper-scale eval is not the current GPU1 job. Official README states
that the full Sudoku reproduction uses 8 GPUs, 423,168 total examples, and took
about 33.6 hours. On the current single-GPU1 constraint, this record first
targets the official quick gate.

## 4. Environment

- GPU: `NVIDIA A800-SXM4-80GB`
- Python/Torch env: `/huyang2/double-loop/official_eqr_compare/.venv`
- Torch: `2.7.0+cu126`
- Attention path: PyTorch SDPA compatibility fallback, because the modern
  FlashAttention wheel is incompatible with this host glibc and the source
  build is blocked by the local CUDA compiler mismatch. This is the same
  runtime fallback that reproduced official Maze exact metrics.
- Compile path: initial compiled launch showed no batch progress after about
  5 minutes and was stopped by exact PID.
- AIStation interruption: the first no-compile run reached `8/16` batches but
  GPU1 was halted by AIStation before metrics were written. It was archived in
  `artifacts/aistation_halt_abort.json` and relaunched after reopening GPU1.
- Successful run: no-compile rerun with the same official config and a unique
  suffix completed in `9599.69s`.

## 5. Commands

```bash
BASE=/huyang2/double-loop/official_eqr_sudoku_repro_20260623
REPO=$BASE/eqr-official
VENV=/huyang2/double-loop/official_eqr_compare/.venv
CKPT=$BASE/downloaded_checkpoints/sudoku-extreme/eqr.pth

cd "$REPO"
source "$VENV/bin/activate"

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled DISABLE_COMPILE=1 \
  python evaluate.py \
  eval_yaml=config/eval/sudoku_lite_noise05.yaml \
  checkpoint="$CKPT" \
  seed=0 \
  suffix="sudoku_lite_D64_B128_N0.5_S1_seed0_rerun_20260623T081821Z" \
  force_rerun=true
```

## 6. Artifacts

- Remote logs:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/logs`
- Remote eval outputs:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/downloaded_checkpoints/eval_preds`
- Remote launcher:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/artifacts/launch_official_sudoku_repro.sh`
- Initial compiled abort metadata:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/artifacts/compiled_abort.json`
- AIStation halt abort metadata:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/artifacts/aistation_halt_abort.json`
- Successful metrics:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/downloaded_checkpoints/eval_preds/step_50000_sudoku_lite_D64_B128_N0.5_S1_seed0_rerun_20260623T081821Z_arch_halt_max_steps-64_arch_noise_scale-0.5_arch_H_init_std-1.0_arch_L_init_std-1.0/eval_metrics_step_50000.json`
- Local archive:
  `runs/official-eqr-sudoku-repro-20260623`

## 7. Results

Official README quick target is reported as a 5-seed mean over 2048 examples
per run. This run is a single seed-0 reproduction of that official quick gate.

| Metric | README quick target | Reproduced seed0 |
|---|---:|---:|
| `convergence_top_k/cumulative_exact_acc_top1` | `99.19 +/- 0.12` | `99.1699` |
| `convergence_top_k/exact_accuracy` | `98.60 +/- 0.04` | `98.6816` |
| `majority_vote/exact_accuracy` | `98.67 +/- 0.05` | `98.7305` |
| `different_init/any_correct` | `99.20 +/- 0.10` | `99.2676` |

Additional diagnostics:

| Metric | Value |
|---|---:|
| `all/exact_accuracy` | `87.9490` |
| `all/accuracy` | `94.9044` |
| `convergence_top_k/total_samples` | `2048` |
| `different_init/n` | `128` |
| eval wall time | `9599.69s` |

## 8. Conclusions

- The official Sudoku-Extreme released-checkpoint quick gate is reproduced on
  GPU1 under official code/data/checkpoint/config. The reproduced seed0 numbers
  are within the README quick target band for the reported primary metrics.
- This validates EqR as a real baseline for the next FutureSeed comparison.
- This is not the full paper-scale 423,168-example reproduction. The official
  README states that full eval used 8 GPUs for about 33.6 hours; under the
  user's GPU1-only constraint, that full reproduction is not an equivalent
  current-budget job.
- The main environment caveat is speed: compatible FlashAttention is blocked on
  this host, so SDPA fallback is numerically adequate but slow. The successful
  seed0 quick run took about `2h40m` on one A800.

## 9. Submission Record

Not applicable.
