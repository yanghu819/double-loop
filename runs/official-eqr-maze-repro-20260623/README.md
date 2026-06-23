# official-eqr-maze-repro-20260623

Official EqR Maze-Unique released-checkpoint reproduction gate.

## Result

| Setting | Paper exact | Reproduced exact | Metric key |
|---|---:|---:|---|
| D16/B1 | 0.822 | 0.827000 | all/exact_accuracy |
| D64/B1 | 0.889 | 0.892000 | all/exact_accuracy |
| D64/B128 | 0.930 | 0.944444 | convergence_top_k/exact_accuracy |

Critical setting: Maze checkpoint reproduction uses `noise_scale=0.01`. The generic `depth_breadth.yaml` default `0.5` collapses exact accuracy while leaving misleading token accuracy.

Artifacts:
- `score.json`
- `remote_logs/logs/`
- `remote_eval_preds/eval_preds/`
- `remote_artifacts/artifacts/sdpa_fallback.patch`
