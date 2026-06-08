# FutureSeed Simple Mechanisms, 2026-06-08

Source SHA: `24a95d5af5a40a532ca03d8083bc95245d11cdf5`

GPU: GPU1 only, NVIDIA A800-SXM4-80GB.

## Question

After clean 16x16 scaling hit an optimization cliff without Sudoku-specific unit-state tricks, test whether a simple FutureSeed mechanism change or generic loop credit assignment can move the frontier.

This was not a table-filling ablation. Each run had a decision:

- FutureSeed EMA decay: does smoothing cross-layer seed state make the 16x16 curriculum viable?
- All-loop supervision: if loops are neutral, is the bottleneck final-loop-only credit assignment?

## Runs

| run | mechanism | train curve | eval result | decision |
|---|---|---:|---:|---|
| `fsdecay16x16-d256-l12-loop8-h32-b16-20260608T0445Z-24a95d5` | launch attempt | no training | n/a | aborted before training because detached worktree lacked local uv |
| `fsdecay16x16-d256-l12-loop8-h32-b16-pybin-20260608T0453Z-24a95d5` | FutureSeed decay=0.5, full planned curriculum | step600 CE buffered to 0.9798 | no eval, killed prematurely | do not treat step500 as final signal |
| `fsdecay16x16-d256-l12-loop8-h32-b16-s600eval-20260608T0520Z-24a95d5` | FutureSeed decay=0.5, total step600 with eval | CE 1.2070 -> 1.0894 -> 1.0231 -> 1.2110 -> 1.2295 -> 0.9798 | h24 loop8 exact 0.0000, blank_acc 0.3354 | decay changes loss curve but not viability |
| `loopall16x16-d256-l12-loop8-h32-b16-s600eval-20260608T0545Z-24a95d5` | FutureSeed decay=0, `LOOP_LOSS=all` | CE 1.2229 -> 1.1263 -> 1.0056 -> 1.2201 -> 1.2404 -> 0.9825 | h24 loop8 exact 0.0000, blank_acc 0.3381 | naive all-loop supervision is not enough |

## Readouts

FutureSeed decay=0.5:

- h16 exact 0.0000, blank_acc 0.3684
- h24 exact 0.0000, blank_acc 0.3354
- h32 exact 0.0000, blank_acc 0.3210
- loop1 to loop8 exact gain: 0.0000

All-loop supervision:

- h16 exact 0.0000, blank_acc 0.3645
- h24 exact 0.0000, blank_acc 0.3381
- h32 exact 0.0000, blank_acc 0.3112
- loop1 to loop8 exact gain: 0.0000

## Insights

1. Step500 was a bad kill boundary for this curriculum. Both completed runs had a delayed CE drop at step600 after looking bad at step500. Future kill rules should require a post-transition window before judging a stage.

2. Simple FutureSeed EMA decay is not the missing mechanism. It reproduced the late CE drop, but h16/h24/h32 exact stayed 0 and loop depth stayed neutral.

3. Naive per-loop supervision is also not the missing mechanism. It slightly improved easy-stage CE at step300, but did not create loop gain or exact solves.

4. Clean 16x16 is still below the per-cell accuracy regime needed for exact Sudoku. This is not the earlier "blank accuracy high but exact zero" global-consistency-only failure; blank_acc around 0.33 means the model is still far from local digit reliability.

5. The practical 80GB clean scale boundary remains D256/L12/loop8/b16 at roughly 42.9GB. Batch32 and batch48 OOM near 79GB.

## Lessons To Reuse

- Detached worktrees should launch with `PYTHON_BIN=/opt/conda/bin/python` and `PYTHON_EXTRA_PATH=/huyang2/double-loop/.cache/python-extra-pylib` when `SKIP_SETUP=1`; otherwise the worktree-local uv bootstrap may be missing.
- Add `PYTHONDONTWRITEBYTECODE=1` for future runs to avoid `__pycache__` making provenance dirty.
- Do not promote decay/gate micro-sweeps unless a simple instance first changes eval behavior.
- For Better Lesson scaling, the next high-ROI move is not another small loss tweak. The bottleneck points to generic capacity/credit-assignment changes that can raise per-cell accuracy first, while avoiding Sudoku-specific row/column/box priors.
