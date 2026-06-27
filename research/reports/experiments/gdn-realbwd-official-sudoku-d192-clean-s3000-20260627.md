# GDN Real-Backward Official Sudoku D192 Clean 3000-Step Scaling

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-clean-s3000-20260627`
- plan ID: `P-GDN-010`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: `e6fe7a03533b0ecb98fa674fd338e89fbf8e0643`
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-27 05:12-05:35 UTC

## 2. Hypothesis

P-GDN-007 showed clean D192/L10 GDN+FutureSeed improves from 600 to 1500 steps. P-GDN-008 showed blind D256 width does not help, and P-GDN-009 showed a small loop-residual FutureSeed memory also does not improve exact. The next high-ROI better-lesson question is whether the clean backbone simply needs to consume more training examples/time.

Prediction: if the current limit is data/optimization budget, a fixed clean D192/L10 FutureSeed backbone should keep improving at 2200/3000 steps over the 1500 baseline (`loop5 exact 0.0293`, `blank_acc 0.5337`). If it stays flat, then same-config longer training is not enough and the next move should be more data/curriculum or a larger clean backbone, not more epochs on this exact config.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
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
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_UPDATE=fixed`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep, no decay/gate sweep.

## 4. Environment

- host: AIStation `GPU1`
- GPU: NVIDIA A100-SXM4-80GB
- device: CUDA only; no CPU smoke
- torch: `2.7.0+cu126`
- remote run worktree: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-clean-s3000-e6fe7a0-20260627`
- git dirty at run start: `false`

## 5. Commands

Launched from GitHub-truth SHA:

```bash
SOURCE_SHA=e6fe7a03533b0ecb98fa674fd338e89fbf8e0643 \
RUN_STAMP=20260627T0512Z \
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_clean_s3000_20260627.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exactly this run and record as failed infrastructure.
- Step100 wall time above 15 minutes: stop; same-config long training is too slow for ROI.
- Checkpoint eval at step1500 is materially worse than the known 1500 baseline and CE is not improving: stop before 3000.
- If step2200 exact/blank are flat relative to step1500 and CE is noisy upward, stop as same-config plateau.

## 6. Artifacts

- local run: `runs/gdn-d192-official-sudoku-fs-clean-s3000-20260627T0512Z-e6fe7a0`
- visual dashboard: `runs/gdn-d192-official-sudoku-fs-clean-s3000-20260627T0512Z-e6fe7a0/visualizations/index.html`
- case HTML: `runs/gdn-d192-official-sudoku-fs-clean-s3000-20260627T0512Z-e6fe7a0/output/futureseed_loop_case_seed52.html`
- result JSON: `runs/gdn-d192-official-sudoku-fs-clean-s3000-20260627T0512Z-e6fe7a0/output/futureseed_loop_seed52.json`
- result MD: `runs/gdn-d192-official-sudoku-fs-clean-s3000-20260627T0512Z-e6fe7a0/output/futureseed_loop_seed52.md`
- checkpoint evals: `runs/gdn-d192-official-sudoku-fs-clean-s3000-20260627T0512Z-e6fe7a0/output/checkpoint_eval_step001500.json`, `runs/gdn-d192-official-sudoku-fs-clean-s3000-20260627T0512Z-e6fe7a0/output/checkpoint_eval_step002200.json`
- score: `runs/gdn-d192-official-sudoku-fs-clean-s3000-20260627T0512Z-e6fe7a0/score.json`
- remote log: `/huyang2/double-loop/artifacts/logs/gdn-d192-official-sudoku-fs-clean-s3000-20260627T0512Z-e6fe7a0.log`

## 7. Results

Primary score: `metrics.eval_clean.loop5.label_exact = 0.02978515625`.

Training curve:

| step | CE | loop1 CE | loop-last CE |
|---:|---:|---:|---:|
| 100 | 1.8010 | 1.8018 | 1.8010 |
| 300 | 1.1538 | 1.2558 | 1.1538 |
| 600 | 1.0566 | 1.2163 | 1.0566 |
| 900 | 1.0370 | 1.2286 | 1.0370 |
| 1200 | 0.9506 | 1.1296 | 0.9506 |
| 1500 | 0.9956 | 1.2129 | 0.9956 |
| 2200 | 0.9835 | 1.1990 | 0.9835 |
| 2800 | 0.9162 | 1.1470 | 0.9162 |
| 3000 | 0.9633 | 1.1972 | 0.9633 |

Checkpoint evals use fixed checkpoint batches, so compare their trend internally:

| checkpoint | loop1 exact | loop5 exact | loop5 blank_acc |
|---:|---:|---:|---:|
| 1500 | 0.0000 | 0.0181 | 0.5246 |
| 2200 | 0.0015 | 0.0181 | 0.5309 |

Final official Sudoku test eval, `eval_n=2048`, seed `1051`:

| loop | exact | blank_acc | early | late |
|---:|---:|---:|---:|---:|
| 1 | 0.0044 | 0.4644 | 0.4681 | 0.4644 |
| 2 | 0.0283 | 0.5133 | 0.5120 | 0.5214 |
| 3 | 0.0288 | 0.5385 | 0.5372 | 0.5418 |
| 4 | 0.0293 | 0.5410 | 0.5393 | 0.5446 |
| 5 | 0.0298 | 0.5418 | 0.5402 | 0.5454 |

Comparison against clean D192/L10 GDN+FutureSeed 1500-step:

- exact: `0.029296875 -> 0.02978515625` (`+0.00048828125`).
- blank accuracy: `0.5337 -> 0.5418` (`+0.0081`).
- train CE: `0.9875 -> 0.9633` (`-0.0242`).
- loop1->loop5 exact gain: `+0.0239 -> +0.0254`, almost unchanged.
- train time: `1003.1s -> 1569.4s` for 1500 vs 3000 steps; cost rose materially for tiny exact gain.

## 8. Conclusions

Decision: mark P-GDN-010 as done, but only as a negative/plateau boundary for same-config long training.

The clean backbone still learns soft cell accuracy after 1500 steps: CE and blank accuracy improve. But hard full-board exact barely moves. Doubling training budget from 1500 to 3000 steps yields only `+0.00049` exact. The checkpoint evals show the same story: blank accuracy improves from step1500 to step2200, while exact is flat.

This is useful evidence for the scaling story. FutureSeed+GDN is real and keeps improving token-level/blank accuracy, but same-config longer training does not unlock global Sudoku consistency. The better-lesson next move should not be another 4500/6000-step repeat at D192/L10. It should be a larger clean backbone, a larger/harder data curriculum, or a generic objective that rewards global consistency without hand-coded Sudoku repair.

Do not run a no-FS 3000-step control: prior no-FS D192/L10 is plateaued at exact `0`, and this experiment asks about the opened clean FS backbone's remaining slope, not whether no-FS remains dead.

## 9. Submission Record

None.
