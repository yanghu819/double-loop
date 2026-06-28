# GDN Real-Backward Official Sudoku D192 Hard Curriculum Scaling

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-hardcurr-s4000-20260628`
- plan ID: `P-GDN-011`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: `d5658e0501faad20133bd85533b307dd8eb9893b`
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-28 14:35-15:12 UTC

## 2. Hypothesis

P-GDN-010 showed that same-config D192/L10 GDN+FutureSeed 1500->3000 training improves CE and blank accuracy but barely moves full-board exact. The official train split is already large and hard (`46-64` blank cells), but the runner previously ignored `HOLE_STAGES` in official-data mode.

Mechanism question: is the exact plateau partly caused by weak hard-stage pressure, where the model keeps learning average blank accuracy but does not spend enough budget on the hardest missing-token regime?

Prediction: if hard-stage data pressure is the missing better-lesson axis, a clean blank-count curriculum should move loop5 exact above the `~0.03` plateau, ideally `>=0.06`, without adding rules or repair. If only CE/blank accuracy improve while exact stays around `0.03`, then data hardness alone is not the current bottleneck.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- data change: official loader now supports blank-count filtering through `HOLE_STAGES`
- curriculum: `46-55:800,56-64:3200`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=5`
- batch: `128`
- steps: `4000`
- eval_n: `2048`
- checkpoint eval: `1500,2500`, no train checkpoint save
- dtype: `bfloat16`
- optimizer: `LR=0.0015`, `WEIGHT_DECAY=0.001`
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_UPDATE=fixed`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep, no gate/decay/loss-weight table.

## 4. Environment

- host: AIStation `GPU1`
- GPU target: A800/A100 80GB class GPU, `CUDA_VISIBLE_DEVICES=0`
- remote run worktree: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-hardcurr-s4000-<sha>-20260628`
- git dirty at run start: must be `false` after detached checkout

## 5. Commands

Launch from GitHub-truth SHA:

```bash
SOURCE_SHA=d5658e0501faad20133bd85533b307dd8eb9893b \
RUN_STAMP=20260628T1435Z \
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_hardcurr_s4000_20260628.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exact PID and record as failed infrastructure.
- Step100 wall time above 15 minutes: stop; curriculum run is too slow for ROI.
- Checkpoint step1500 has CE much worse than the 3000-step clean run and loop5 exact still zero: stop as optimization failure.
- Checkpoint step2500 shows blank accuracy improving but exact flat near the old plateau: continue only if CE slope is still strong; otherwise stop and archive as data-curriculum boundary.

## 6. Artifacts

- local run: `runs/gdn-d192-official-sudoku-fs-hardcurr-s4000-20260628T1435Z-d5658e0`
- visual dashboard: `runs/gdn-d192-official-sudoku-fs-hardcurr-s4000-20260628T1435Z-d5658e0/visualizations/index.html`
- case HTML: `runs/gdn-d192-official-sudoku-fs-hardcurr-s4000-20260628T1435Z-d5658e0/output/futureseed_loop_case_seed52.html`
- result JSON: `runs/gdn-d192-official-sudoku-fs-hardcurr-s4000-20260628T1435Z-d5658e0/output/futureseed_loop_seed52.json`
- result MD: `runs/gdn-d192-official-sudoku-fs-hardcurr-s4000-20260628T1435Z-d5658e0/output/futureseed_loop_seed52.md`
- checkpoint evals: `runs/gdn-d192-official-sudoku-fs-hardcurr-s4000-20260628T1435Z-d5658e0/output/checkpoint_eval_step001500.json`, `runs/gdn-d192-official-sudoku-fs-hardcurr-s4000-20260628T1435Z-d5658e0/output/checkpoint_eval_step002500.json`
- score: `runs/gdn-d192-official-sudoku-fs-hardcurr-s4000-20260628T1435Z-d5658e0/score.json`
- remote log: `/huyang2/double-loop/artifacts/logs/gdn-d192-official-sudoku-fs-hardcurr-s4000-20260628T1435Z-d5658e0.log`

## 7. Results

Primary score: `metrics.eval_clean.loop5.label_exact = 0.03125`.

Official data distribution:

| split | rows | blank min | blank max | blank mean |
|---|---:|---:|---:|---:|
| train | 1,001,000 | 46 | 64 | 55.7670 |
| test | 422,786 | 45 | 64 | 55.7930 |

Training curve:

| step | stage | CE | loop1 CE | loop-last CE |
|---:|---|---:|---:|---:|
| 100 | 46-55 | 1.7786 | 1.7791 | 1.7786 |
| 300 | 46-55 | 1.0411 | 1.1026 | 1.0411 |
| 600 | 46-55 | 0.9464 | 1.1313 | 0.9464 |
| 800 | 46-55 | 0.9693 | 1.1876 | 0.9693 |
| 1200 | 56-64 | 1.0367 | 1.2218 | 1.0367 |
| 1500 | 56-64 | 1.0690 | 1.2507 | 1.0690 |
| 2200 | 56-64 | 1.0229 | 1.2190 | 1.0229 |
| 2500 | 56-64 | 1.0569 | 1.2272 | 1.0569 |
| 3000 | 56-64 | 1.0367 | 1.2600 | 1.0367 |
| 3500 | 56-64 | 1.0021 | 1.2675 | 1.0021 |
| 4000 | 56-64 | 0.9903 | 1.2650 | 0.9903 |

Checkpoint evals use fixed checkpoint batches, so compare their trend internally:

| checkpoint | loop1 exact | loop5 exact | loop5 blank_acc |
|---:|---:|---:|---:|
| 1500 | 0.0000 | 0.0181 | 0.5199 |
| 2500 | 0.0010 | 0.0181 | 0.5345 |

Final official Sudoku test eval, `eval_n=2048`, seed `1051`:

| loop | exact | blank_acc | early | late |
|---:|---:|---:|---:|---:|
| 1 | 0.0029 | 0.4503 | 0.4531 | 0.4511 |
| 2 | 0.0278 | 0.4997 | 0.4997 | 0.5009 |
| 3 | 0.0293 | 0.5392 | 0.5366 | 0.5428 |
| 4 | 0.0303 | 0.5454 | 0.5438 | 0.5482 |
| 5 | 0.0312 | 0.5462 | 0.5443 | 0.5490 |

Comparison against clean D192/L10 GDN+FutureSeed 3000-step:

- exact: `0.02978515625 -> 0.03125` (`+0.00146484375`).
- blank accuracy: `0.5418 -> 0.5462` (`+0.0044`).
- train CE: `0.9633 -> 0.9903`, worse despite 4000 steps because the hard stage dominates training.
- loop1->loop5 exact gain: `+0.0254 -> +0.0283`, a small improvement but still the same plateau.
- checkpoint exact stayed flat from step1500 to step2500 while blank accuracy improved.
- train time: `2703.2s`, peak allocated memory `48031MB`.

## 8. Conclusions

Decision: mark P-GDN-011 as discarded/negative for opening the hard exact plateau.

The implementation change is worth keeping: official-data `HOLE_STAGES` now does what the command line says, and the run cleanly records train/eval blank distributions. But the experiment does not support continuing this exact hard-curriculum axis.

The result is the same failure shape as P-GDN-010. The model learns more soft cell accuracy and later loops still add some refinement, but full-board exact barely moves. Forcing 56-64 blank examples for most of training raises loop5 exact only from `0.0298` to `0.03125`, far below the `>=0.06` success bar. The checkpoint eval is even clearer: loop5 exact is flat at `0.0181` from step1500 to step2500 while blank accuracy rises.

Insight: hard-data pressure alone is not enough. The current bottleneck is not merely that the model sees too few hard official samples. It is still global consistency: many cells get locally better, but the probability of the whole board being correct does not compound fast enough.

Next high-ROI direction should not be another D192 hard-curriculum repeat or a blank-count weight sweep. The next move needs either a different scalable capacity axis or a generic global-consistency/state objective that remains rule-free.

## 9. Submission Record

None.
