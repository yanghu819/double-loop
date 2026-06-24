# official-eqr-mixer-replacement-triton-e1024-matched-eval-20260624T1015Z-ce2e76b

Matched official EqR Sudoku tiny eval gate after e1024 train-only runs.

- Machine: AIStation `GPU1` only
- Source SHA: `ce2e76bed92d2f271518f0bf23f251c8eec492e5`
- Eval config: `/huyang2/double-loop/artifacts/eval_configs/sudoku_tiny_d16_n001.yaml`
- Eval budget: `max_eval_steps=16`, batch 128, `halt_max_steps=16`, `noise_scale=0.01`, `different_init=1`
- Base checkpoint: `/huyang2/double-loop/official_eqr_compare/outputs/mixer-base/outputs/y9l15z0t/2026-6-24/9-53-24/checkpoints/step_7168_y9l15z0t.pth`
- FutureSeed checkpoint: `/huyang2/double-loop/official_eqr_compare/outputs/futureseed-mixer/outputs/d8w7plle/2026-6-24/9-59-30/checkpoints/step_7168_d8w7plle.pth`

## Result

| arm | train loss | accuracy | exact | lm_loss | total_loss | residual1 | residual4 | residual8 | residual16 | eval sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official mixer-base | 0.767679 | 0.664400 | 0.024902 | 0.766357 | 0.791852 | 195.923 | 4.833 | 4.849 | 4.979 | 7.957 |
| FutureSeed mixer | 1.408197 | 0.423050 | 0.000000 | 1.402906 | 1.402906 | 198.346 | 6.381 | 3.238 | 3.065 | 14.814 |
| FS - base | +0.640518 | -0.241350 | -0.024902 | +0.636548 | +0.611054 | +2.422 | +1.548 | -1.611 | -1.914 | +6.857 |

## Decision

Stop simply lengthening the same FutureSeed mixer replacement. It wins early, but by e1024 the official mixer-base clearly opens exact and reaches much better accuracy/loss. The lower late-loop residual for FutureSeed is not a win here; it means the replacement reaches a stable but wrong attractor.
