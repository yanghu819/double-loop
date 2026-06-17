# Maze31 Cross-Candidate State Competition

Run: `maze31-statecompete-cross-d256l4-s800-20260617T164053Z-f19e66a`

Source SHA: `f19e66abb0714b4021ae84423ab7ca1f283ca63f`

UTC launch: `2026-06-17T16:40:53Z`

## Question

Can a stronger generic context candidate make loop-state competition produce real pruning, instead of collapsing onto the proposed state?

The previous `state_compete` and `final-only` probes showed that the model mostly copies a broad high-recall path mask through all loops. The proposed candidate dominated the softmax at about `0.96`, so keep/context barely participated.

## Mechanism

This run used the new `state_compete_cross` mode. It keeps the same three candidates, but changes the context candidate:

```text
cross_features = [previous, proposed, proposed - previous, previous * proposed]
context = noncausal_attention(proposed + linear(cross_features))
next = weighted_sum(keep, proposed, context)
```

This is generic sequence/state modeling. It does not encode maze topology, shortest-path rules, repair, or a selector.

The competition logits are initialized neutral in this mode so the alternative candidate receives gradient early. This is part of the anti-collapse mechanism, not a sweep.

## Prediction

Positive evidence would be:

- loop12 precision higher than loop1,
- loop12 predicted path fraction lower than loop1,
- non-collapsed candidate weights,
- visible false-positive branch pruning in the casebook.

If candidate weights diversify but the loop trajectory stays flat, then candidate diversity alone is not enough; the alternative candidate is different but not carrying error-correcting information.

## Budget And Kill Criteria

- GPU: GPU1 only, `CUDA_VISIBLE_DEVICES=0`.
- Config: Maze31 perfect mazes, path length `160-260`, D256/L4, `batch=16`, `train_loops=6`, `eval_loops=12`, `loop_loss=all`.
- Budget: one 800-step probe.
- Kill criteria: stop if step100 exceeded 15 minutes, if step300 path F1 stayed below `0.50`, or if GPU utilization/resource state was bad.

The run reached step800 in `630.8s`, so it was allowed to finish.

## Result

Primary: loop12 path F1 `0.5847`, exact `0.0000`, loop gain `+0.0002`.

| loop | path_f1 | precision | recall | pred_frac | keep | proposed | context | context-proposed RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5845 | 0.4156 | 0.9964 | 0.4632 | 0.1476 | 0.4938 | 0.3586 | 0.8265 |
| 2 | 0.5846 | 0.4159 | 0.9958 | 0.4626 | 0.3642 | 0.3897 | 0.2461 | 1.0227 |
| 4 | 0.5847 | 0.4160 | 0.9956 | 0.4624 | 0.3723 | 0.3873 | 0.2404 | 1.0324 |
| 8 | 0.5847 | 0.4160 | 0.9953 | 0.4622 | 0.3705 | 0.3886 | 0.2409 | 1.0323 |
| 12 | 0.5847 | 0.4160 | 0.9953 | 0.4622 | 0.3704 | 0.3887 | 0.2409 | 1.0323 |

## Visual Failure Shape

The candidate collapse is fixed, but the output still does not become a correction process. Case 319 is representative: loop1 has `288` false positives and `2` misses; loop12 has `288` false positives and `3` misses. The false-positive path branches remain.

Artifacts:

- `output/eqr_maze_probe_fs1_seed52.json`
- `output/eqr_maze_probe_fs1_seed52.md`
- `output/visualizations/index.html`
- `output/visualizations/casebook.md`

## Decision

This is a useful negative result.

`state_compete_cross` proves that we can make the candidates genuinely different and keep weights from collapsing. However, loop gain remains near zero. Therefore the current bottleneck is no longer just softmax collapse or candidate diversity. The alternative state must be trained or structured to carry error-correcting information across loops.

Next high-ROI direction: make the alternative candidate predictive of a future corrected state, not merely different from proposed. A simple version would be a self-supervised temporal target inside the loop, for example encouraging an early-loop context candidate to match a later-loop state/logit direction. That still follows the bitter lesson: no maze repair, no hand-coded graph search, just scalable learned dynamics.
