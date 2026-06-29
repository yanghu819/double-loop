# GDN Real-Backward Official Sudoku D192 Exact-Margin Probe

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-exactmargin-s3000-20260629`
- plan ID: `P-GDN-012`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: `6dde82a4aaa82eed2ff91a8f079dc5eb5fe066a5`
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-29 02:51-03:15 UTC

## 2. Hypothesis

P-GDN-010 and P-GDN-011 both show the same failure shape: training CE and blank accuracy keep improving, while full-board exact stays around `0.03`. That suggests the objective may be optimizing average cell quality while each board is still killed by a few weakest blank cells.

Mechanism question: can a generic per-sample weakest-blank-cell margin objective move exact, without adding Sudoku rules or postprocessing?

Prediction: if exact is limited by average CE under-training the weakest cells, then pushing the true-logit versus best-other-logit soft-min margin should improve loop5 exact more than blank accuracy alone. If the margin diagnostic improves but exact remains flat, the bottleneck is not this objective mismatch; it is more likely model/state capacity or a deeper global-consistency issue.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- training blank-count range: `45-64`, the full official train/test range rather than a hard curriculum
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=5`
- batch: `128`
- steps: `3000`
- eval_n: `2048`
- checkpoint eval: `1500,2200`, no train checkpoint save
- dtype: `bfloat16`
- optimizer: `LR=0.0015`, `WEIGHT_DECAY=0.001`
- blank CE weight: `8`
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_UPDATE=fixed`
- exact-margin objective: `EXACT_MARGIN_WEIGHT=0.2`, `EXACT_MARGIN_TAU=0.5`, `EXACT_MARGIN_TARGET=0.0`, `EXACT_MARGIN_START_STEP=800`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep, no loss-weight table.

## 4. Environment

- host: AIStation `GPU1`
- GPU target: A800/A100 80GB class GPU, `CUDA_VISIBLE_DEVICES=0`
- remote run worktree: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-exactmargin-s3000-<sha>-20260629`
- git dirty at run start: must be `false` after detached checkout

## 5. Commands

Launch from GitHub-truth SHA:

```bash
SOURCE_SHA=<launch-sha> \
RUN_STAMP=<utc-stamp> \
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_exactmargin_s3000_20260629.sh
```

Actual launch:

```bash
SOURCE_SHA=6dde82a4aaa82eed2ff91a8f079dc5eb5fe066a5 \
RUN_STAMP=20260629T0251Z \
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_exactmargin_s3000_20260629.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exact PID and record infrastructure failure.
- Step100 wall time above 15 minutes: stop; the probe is too slow for ROI.
- After margin starts, if CE jumps above `1.3` and checkpoint exact remains zero, stop as optimization instability.
- If checkpoint step1500 and step2200 show exact flat near old baseline while exact-margin softmin improves, finish only if runtime is still cheap enough; otherwise archive as objective-mismatch boundary.

Success criteria:

- Strong: loop5 exact `>=0.06`.
- Useful: loop5 exact improves by at least `+0.01` over clean 3000 baseline `0.0298`, without blank accuracy collapse.
- Negative: blank accuracy or margin improves but exact stays around `0.03`.

## 6. Artifacts

- local run: `runs/gdn-d192-official-sudoku-fs-exactmargin-s3000-20260629T0251Z-6dde82a`
- visual dashboard: `runs/gdn-d192-official-sudoku-fs-exactmargin-s3000-20260629T0251Z-6dde82a/visualizations/index.html`
- case HTML: `runs/gdn-d192-official-sudoku-fs-exactmargin-s3000-20260629T0251Z-6dde82a/output/futureseed_loop_case_seed52.html`
- result JSON: `runs/gdn-d192-official-sudoku-fs-exactmargin-s3000-20260629T0251Z-6dde82a/output/futureseed_loop_seed52.json`
- result MD: `runs/gdn-d192-official-sudoku-fs-exactmargin-s3000-20260629T0251Z-6dde82a/output/futureseed_loop_seed52.md`
- checkpoint evals: `runs/gdn-d192-official-sudoku-fs-exactmargin-s3000-20260629T0251Z-6dde82a/output/checkpoint_eval_step001500.json`, `runs/gdn-d192-official-sudoku-fs-exactmargin-s3000-20260629T0251Z-6dde82a/output/checkpoint_eval_step002200.json`
- score: `runs/gdn-d192-official-sudoku-fs-exactmargin-s3000-20260629T0251Z-6dde82a/score.json`
- source snapshot: `runs/gdn-d192-official-sudoku-fs-exactmargin-s3000-20260629T0251Z-6dde82a/source_snapshot.tar.gz`
- remote log: `/huyang2/double-loop/artifacts/logs/gdn-d192-official-sudoku-fs-exactmargin-s3000-20260629T0251Z-6dde82a.log`

## 7. Results

Primary score: `metrics.eval_clean.loop5.label_exact = 0.029296875`.

Training curve:

| step | CE | total | exact-margin loss | softmin margin |
|---:|---:|---:|---:|---:|
| 100 | 1.8010 | 1.8010 | 0.0000 | 0.000 |
| 300 | 1.1538 | 1.1538 | 0.0000 | 0.000 |
| 600 | 1.0566 | 1.0566 | 0.0000 | 0.000 |
| 800 | 1.0285 | 1.3058 | 1.3866 | -0.967 |
| 900 | 1.0724 | 1.2529 | 0.9025 | -0.367 |
| 1200 | 0.9806 | 1.1584 | 0.8888 | -0.260 |
| 1500 | 1.0233 | 1.2015 | 0.8912 | -0.313 |
| 2000 | 0.9956 | 1.1618 | 0.8308 | -0.128 |
| 2200 | 1.0062 | 1.1768 | 0.8530 | -0.221 |
| 2600 | 0.9917 | 1.1668 | 0.8754 | -0.251 |
| 2800 | 0.9414 | 1.1109 | 0.8471 | -0.094 |
| 3000 | 0.9902 | 1.1556 | 0.8269 | -0.129 |

Checkpoint evals use fixed checkpoint batches, so compare their trend internally:

| checkpoint | loop5 exact | loop5 blank_acc | train CE | exact-margin loss | softmin |
|---:|---:|---:|---:|---:|---:|
| 1500 | 0.0181 | 0.5206 | 1.0233 | 0.8912 | -0.3127 |
| 2200 | 0.0186 | 0.5269 | 1.0062 | 0.8530 | -0.2211 |

Final official Sudoku test eval, `eval_n=2048`, seed `1051`:

| loop | exact | blank_acc | early | late |
|---:|---:|---:|---:|---:|
| 1 | 0.0020 | 0.4436 | 0.4436 | 0.4370 |
| 2 | 0.0273 | 0.5022 | 0.4972 | 0.5069 |
| 3 | 0.0293 | 0.5312 | 0.5293 | 0.5343 |
| 4 | 0.0293 | 0.5358 | 0.5346 | 0.5387 |
| 5 | 0.0293 | 0.5362 | 0.5348 | 0.5388 |

Comparison against clean D192/L10 GDN+FutureSeed 3000-step:

- exact: `0.02978515625 -> 0.029296875` (`-0.00048828125`).
- blank accuracy: `0.5418 -> 0.5362` (`-0.0056`).
- train CE: `0.9633 -> 0.9902`, worse because margin pressure competes with average CE.
- train time: `1569.4s -> 1745.1s`, about `+11.2%`.
- checkpoint 1500 exact: both `0.0181`; blank `0.5246 -> 0.5206`.
- checkpoint 2200 exact: `0.0181 -> 0.0186`; blank `0.5309 -> 0.5269`.
- peak allocated memory: unchanged at about `48031MB`.

The objective did optimize its own diagnostic after step800: softmin moved from `-0.967` at step800 to about `-0.129` by step3000. That improvement did not translate into full-board exact or blank accuracy.

## 8. Conclusions

Decision: mark P-GDN-012 as discarded.

This is a useful negative result. The exact-margin loss is trainable and does not destabilize the GPU run, but it does not solve the plateau. It makes the model spend capacity on weakest blank-cell margins, while final exact and blank accuracy are slightly worse than the clean 3000-step baseline.

Insight: the current bottleneck is not simply that average CE ignores the weakest few cells. A generic per-sample margin target can improve a margin diagnostic without producing more globally correct Sudoku boards. This points back to model/state capacity or a more structural but still generic state dynamics problem, not another loss-weight sweep.

Do not sweep `EXACT_MARGIN_WEIGHT`, `TAU`, or `TARGET`. The high-ROI next step should be a stronger scaling/state axis that remains bitter-lesson-compliant, for example larger recurrent state/capacity with enough data, or a generic state update that changes how loops accumulate and correct information rather than adding another indirect regularizer.

Implementation lesson: the background start wrapper wrote a PID file, but the inner launch script deleted it via `rm -f "$PID"`. The run was still monitored by exact PID `229`; the launch script should stop deleting the wrapper PID file before future runs.

## 9. Submission Record

None.
