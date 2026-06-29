# GDN Real-Backward Official Sudoku D192 Exact-Margin Probe

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-exactmargin-s3000-20260629`
- plan ID: `P-GDN-012`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: pending launch SHA
- local branch: `codex/gpu1-experiment-tracking`
- planned launch time: 2026-06-29 UTC

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

Pending launch.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

None.
