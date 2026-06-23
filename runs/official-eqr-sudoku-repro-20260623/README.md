# official-eqr-sudoku-repro-20260623

Status: done.

This run is the official EqR Sudoku-Extreme released-checkpoint reproduction
gate on AIStation GPU1. It uses upstream EqR SHA
`aba94e9cde0f273ce644db5261cd6915ba6561f0`, official
`sudoku-extreme-1k-aug-1000` data, official
`locuslab/EqR-model/sudoku-extreme/eqr.pth`, and
`config/eval/sudoku_lite_noise05.yaml`.

The quick gate settings are official D64/B128 Sudoku eval:
`halt_max_steps=64`, `different_init=128`, `convergence_top_k=4`,
`noise_scale=0.5`, `init_std=1.0`, `global_batch_size=128`, and
`max_eval_steps=16`.

The reproduced seed0 quick metrics are:

| Metric | Reproduced |
|---|---:|
| `convergence_top_k/cumulative_exact_acc_top1` | `0.991699` |
| `convergence_top_k/exact_accuracy` | `0.986816` |
| `majority_vote/exact_accuracy` | `0.987305` |
| `different_init/any_correct` | `0.992676` |

These match the official quick gate target range from the EqR README. The full
paper-scale 423,168-example eval remains a separate 8-GPU, about-33.6-hour job
and was not run under the GPU1-only constraint.

Current caveat: the host cannot load a compatible modern FlashAttention wheel,
so this uses the same PyTorch SDPA runtime fallback that reproduced the official
Maze released-checkpoint numbers. The successful seed0 quick run took
`9599.69s` on one A800.

Artifacts:

- `eval_metrics_step_50000.json`
- `eval_metrics_step_50000.csv`
- `eval_config.yaml`
- `logs/`
- `artifacts/compiled_abort.json`
- `artifacts/aistation_halt_abort.json`
- `official_eqr_sudoku_repro_metadata_20260623T110243Z.tgz`
