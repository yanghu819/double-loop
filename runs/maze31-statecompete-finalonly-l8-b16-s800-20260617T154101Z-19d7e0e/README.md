# Maze31 State Competition Final-Only Recurrence Pressure

Run: `maze31-statecompete-finalonly-l8-b16-s800-20260617T154101Z-19d7e0e`

Source SHA: `19d7e0eda7db61fc002e4cfc7cfeb1345181c191`

UTC launch: `2026-06-17T15:41:01Z`

## Question

Can recurrence pressure prevent `state_compete` from becoming a one-step broad-mask predictor?

The previous `state_compete` run had a small pruning-direction signal versus readout-gate, but loop12 still put about `96%` weight on the proposed candidate and loop gain was effectively zero. This probe changed the training pressure, not the mechanism: supervise only the final recurrent loop and train with 8 loops, so the model should have a reason to use later loops instead of making loop1 solve everything.

## Prediction

Positive evidence would be loop12 becoming narrower or more precise than loop1:

- higher loop12 precision,
- lower loop12 predicted path fraction,
- non-collapsed keep/context candidate weights,
- visible false-positive branch removal in the casebook.

A pure F1 tie or a broader high-recall mask would count against this recurrence-pressure route.

## Budget And Kill Criteria

- GPU: GPU1 only, `CUDA_VISIBLE_DEVICES=0`.
- Config: Maze31 perfect mazes, path length `160-260`, D256/L4, `train_loops=8`, `eval_loops=12`, `loop_loss=final`, `state_update_mode=state_compete`.
- Initial `batch=32` OOMed on A800 80GB and was archived separately with `abort.json`.
- Relaunch used `batch=16`, same mechanism and same hypothesis.
- Kill criteria: stop if step100 exceeded 15 minutes, or if step300 showed dead learning. The run reached step800 in `492.3s`, so it was allowed to finish.

## Result

Primary score: loop12 path F1 `0.5852`, exact `0.0000`, loop gain `-0.0001`.

| loop | path_f1 | precision | recall | pred_frac | keep | proposed | context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5853 | 0.4163 | 0.9968 | 0.4626 | 0.0186 | 0.9615 | 0.0199 |
| 2 | 0.5851 | 0.4161 | 0.9972 | 0.4630 | 0.0195 | 0.9606 | 0.0199 |
| 4 | 0.5851 | 0.4162 | 0.9970 | 0.4629 | 0.0195 | 0.9606 | 0.0199 |
| 8 | 0.5851 | 0.4162 | 0.9970 | 0.4629 | 0.0195 | 0.9606 | 0.0199 |
| 12 | 0.5852 | 0.4162 | 0.9970 | 0.4629 | 0.0195 | 0.9606 | 0.0199 |

## Visual Failure Shape

The casebook shows almost no qualitative trajectory from loop1 to loop12. Case 390 is representative: loop1 has `287` false positives and `3` misses; loop12 still has `287` false positives and `3` misses. The model draws a wide connected path hypothesis and then repeats it.

This is not a search process slowly correcting itself. It is a stable high-recall mask.

Artifacts:

- `output/eqr_maze_probe_fs1_seed52.json`
- `output/eqr_maze_probe_fs1_seed52.md`
- `output/visualizations/index.html`
- `output/visualizations/casebook.md`

## Decision

This is a high-information negative result for simple recurrence pressure. Final-only supervision and more training loops do not make keep/context candidates carry real comparison work.

The next useful experiment is not a seed, bias, or temperature sweep. It should strengthen the candidate-generation side: the context candidate must become an actual alternative path hypothesis that can compete with the proposed state, while staying generic and not encoding maze rules.

Paper-story implication: the thesis still holds, but this probe localizes the current bottleneck. FutureSeed gives a broad direction and loop provides compute, yet state dynamics only becomes meaningful if later states contain competing hypotheses. A scalar or softmax over weak candidates cannot create pruning by itself.
