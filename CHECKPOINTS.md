# Research Checkpoints

This repository keeps one active research checkpoint and one maintained code
surface.

## Active

| checkpoint | commit | why it matters |
| --- | --- | --- |
| `checkpoint/clean-feature-diff-main-2026-06-01` | tag target | Final maintained mainline: RWKV + FutureSeed + depth loop + feature-difference stochastic rollout only. |

## Maintained Code

Only this codebase is maintained:

- `experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py`

It intentionally has no GRU/attention fallback, no Sudoku rule helper, no
learned/posterior guidance branch, and no repeat-view Sudoku representation.

## Active Run Artifacts

Only the best positive run directory stays under `experiments/rwkv_fs_sudoku/runs/`:

- `final_feature_diff_main_6x6`: best feature-difference rollout result.
