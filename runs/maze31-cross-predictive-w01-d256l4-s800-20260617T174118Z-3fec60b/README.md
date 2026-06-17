# Maze31 Cross-Candidate Predictive State Objective

Run: `maze31-cross-predictive-w01-d256l4-s800-20260617T174118Z-3fec60b`

Source SHA: `3fec60b04ae6a69cdbbf5a2b7c3e04ef7a72a153`

UTC launch: `2026-06-17T17:41:18Z`

## Question

Can the alternative context candidate learn a useful correction direction if it is trained to predict a future recurrent state/logit?

The previous `state_compete_cross` run fixed candidate collapse: loop12 used keep/proposed/context around `0.370/0.389/0.241`, and the context candidate was genuinely different from proposed. But loop gain remained near zero. This probe asked whether a simple temporal prediction target can turn that different candidate into a useful future-state predictor.

## Mechanism

For every training rollout, the context candidate at loop `t` is read through the shared `lm_head`, producing `context_logits[t]`. The auxiliary loss is:

```text
MSE(context_logits[t], stop_gradient(logits[t + 1]))
```

This is a generic temporal prediction objective. It does not encode maze rules, graph search, repair, or selector logic. It only asks the alternative candidate to predict the model's next recurrent output.

Config:

- `state_update_mode=state_compete_cross`
- `predictive_state_weight=0.1`
- `predictive_state_horizon=1`

## Prediction

Positive evidence would be:

- predictive loss decreases,
- loop12 precision rises versus loop1,
- loop12 predicted path fraction falls versus loop1,
- candidate weights stay diverse,
- casebook shows false-positive branch removal.

If predictive loss decreases but loop outputs stay flat, then naive next-step imitation is not enough; the target must be improvement-aware rather than just future-state copying.

## Budget And Kill Criteria

- GPU: GPU1 only, `CUDA_VISIBLE_DEVICES=0`.
- Config: Maze31 perfect mazes, path length `160-260`, D256/L4, `batch=16`, `train_loops=6`, `eval_loops=12`, `loop_loss=all`.
- Budget: one 800-step probe.
- Kill criteria: stop if step100 exceeded 15 minutes, if step300 path F1 stayed below `0.50`, or if GPU/resource state was bad.

The run reached step800 in `626.7s`, so it was allowed to finish.

## Result

Primary: loop12 path F1 `0.5837`, exact `0.0000`, loop gain approximately `0.0000`.

The predictive objective was learned:

| step | predictive loss | train path_f1 |
| --- | ---: | ---: |
| 100 | 0.0094 | 0.5797 |
| 200 | 0.0069 | 0.5665 |
| 300 | 0.0043 | 0.5849 |
| 400 | 0.0049 | 0.5856 |
| 500 | 0.0036 | 0.5600 |
| 600 | 0.0026 | 0.5625 |
| 700 | 0.0020 | 0.5809 |
| 800 | 0.0018 | 0.5773 |

But the loop output did not improve:

| loop | path_f1 | precision | recall | pred_frac | keep | proposed | context | next-logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5837 | 0.4142 | 1.0000 | 0.4664 | 0.0533 | 0.3868 | 0.5599 | 0.0047 |
| 2 | 0.5837 | 0.4142 | 1.0000 | 0.4664 | 0.3967 | 0.2556 | 0.3477 | 0.0011 |
| 4 | 0.5837 | 0.4142 | 1.0000 | 0.4664 | 0.3916 | 0.2590 | 0.3495 | 0.0009 |
| 8 | 0.5837 | 0.4142 | 1.0000 | 0.4664 | 0.3881 | 0.2607 | 0.3514 | 0.0010 |
| 12 | 0.5837 | 0.4142 | 1.0000 | 0.4664 | 0.3880 | 0.2607 | 0.3514 | n/a |

## Visual Failure Shape

Case 62 is representative. Loop1 has `290` false positives and `0` misses. Loop12 still has `290` false positives and `0` misses. The model preserves a full-recall broad mask; it does not prune.

Artifacts:

- `output/eqr_maze_probe_fs1_seed52.json`
- `output/eqr_maze_probe_fs1_seed52.md`
- `output/visualizations/index.html`
- `output/visualizations/casebook.md`

## Decision

This is a useful negative result.

The objective is learnable, but it teaches self-copying of the current broad mask. It does not create correction. The next target cannot be plain next-step prediction; it must be improvement-aware. The cleanest next direction is a contrastive or residual predictive target, for example making the context candidate predict a later-loop improvement direction over loop1 rather than the next logits themselves.

Paper-story implication: FutureSeed can give a broad direction, loop gives compute, and candidate state dynamics can avoid collapse. But sustained correction needs a training signal that distinguishes "future" from "better", otherwise recurrence learns to preserve its first broad hypothesis.
