# Maze31 State Competition Final-Only B32 Abort

Run: `maze31-statecompete-finalonly-l8-s800-20260617T153917Z-19d7e0e`

Source SHA: `19d7e0eda7db61fc002e4cfc7cfeb1345181c191`

UTC launch: `2026-06-17T15:39:17Z`

## Intent

This was the first resource attempt for the recurrence-pressure probe: same Maze31 D256/L4 `state_compete` mechanism, `train_loops=8`, `eval_loops=12`, and `loop_loss=final`, with `MAZE_BATCH=32`.

## Abort

The run failed immediately with CUDA OOM on the A800 80GB card. The error happened before step100, during the first training pass.

The run was not used for model-quality conclusions. It only established the resource boundary for this exact probe: D256/L4 with 8 differentiable train loops does not fit batch 32 on 80GB.

## Follow-Up

The same mechanism and hypothesis were relaunched as `maze31-statecompete-finalonly-l8-b16-s800-20260617T154101Z-19d7e0e` with `MAZE_BATCH=16`. That is a resource adaptation, not an ablation.

Artifacts:

- `abort.json`
- `logs/run.log`
- `config.json`
