# GDN Real-Backward Official Sudoku D192 Feedback-Attractor Probe

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-feedback-attractor-s1500-20260629`
- plan ID: `P-GDN-015`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: `4dfba713ba427b8b8ab83cd9fea072de2a480645`
- local branch: `codex/gpu1-experiment-tracking`
- planned: 2026-06-29 07:08 UTC
- launched: 2026-06-29 09:19 UTC after GPU1 queue cleared
- recorded: 2026-06-29 09:36 UTC

## 2. Hypothesis

P-GDN-013 and P-GDN-014 showed the current bottleneck clearly: FutureSeed+GDN opens official Sudoku and loops 1->3 do real work, but later loops copy the same operating point. EqR's useful lesson is not "copy the mixer"; it is "train the recurrent process as an attractor that can recover from bad intermediate states."

Mechanism question: if the next loop sees a corrupted version of the previous loop prediction through a learned generic feedback channel, can FutureSeed+loop learn to pull state back toward the correct board and keep reducing wrong cells after loop3?

Prediction: if the missing piece is attractor-style state training, feedback corruption should make loop5 visibly more corrective than the clean fixed-state baseline: exact should break above the `0.0293` plateau or official hard-case wrong counts should keep dropping from loop3 to loop5. If feedback/corruption diagnostics are active but exact and hard cases remain flat, this specific attractor pressure is not sufficient.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- training blank-count range: full official train split, `45-64`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=5`
- batch: `128`
- steps: `1500`
- eval_n: `2048`
- checkpoint eval: `800,1200`, no train checkpoint save
- case bank: official eval fixed batch, `CASE_BANK_N=3`, `CASE_BANK_EVAL_N=256`, loops `1,2,3,5`
- dtype: `bfloat16`
- optimizer: `LR=0.0015`, `WEIGHT_DECAY=0.001`
- blank CE weight: `8`
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_UPDATE=fixed`
- feedback: `LOOP_FEEDBACK_SCALE=1.0`, detached previous probabilities
- attractor pressure: train-time feedback corruption with `LOOP_FEEDBACK_CORRUPT_PROB=0.35`, `LOOP_FEEDBACK_CORRUPT_MIX=0.7`, mode `random_token`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule in training, no seed sweep, no loss table.

## 4. Environment

- host: AIStation `GPU1`
- GPU target: A100 80GB, `CUDA_VISIBLE_DEVICES=0`
- remote run worktree: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-feedback-attractor-s1500-<sha>-20260629`
- git dirty at run start: must be `false` after detached checkout

## 5. Commands

Launch from GitHub-truth SHA:

```bash
SOURCE_SHA=<filled-after-commit> \
RUN_STAMP=20260629T0708Z \
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_feedback_attractor_s1500_20260629.sh
```

Actual launch:

```bash
SOURCE_SHA=4dfba713ba427b8b8ab83cd9fea072de2a480645 \
RUN_STAMP=20260629T0708Z \
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_feedback_attractor_s1500_20260629.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exact PID and record infrastructure failure.
- Step100 wall time above 12 minutes: feedback path is too slow at this batch; relaunch only with lower microbatch if no source change is needed.
- If checkpoint800 has CE above `1.5` and loop5 exact is still zero with blank_acc below `0.45`, stop as optimization failure.
- If feedback diagnostics show `loop_feedback_next_norm` near zero through step300, stop and inspect initialization/training before burning the full budget.

Success criteria:

- Strong: loop5 exact `>=0.05`.
- Useful: loop5 exact improves by at least `+0.01` over the clean D192/L10 1500 baseline (`0.0293`), or hard-case loop3->loop5 wrong counts keep decreasing without true-cell collapse.
- Negative: feedback/corruption are active but exact remains around `0.03` and loop3->loop5 hard cases remain flat.

## 6. Artifacts

- run root: `runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71`
- config: `runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71/config.json`
- score: `runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71/score.json`
- log: `runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71/logs/run.log`
- result JSON/MD/HTML: `runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71/output/futureseed_loop_seed52.{json,md,html}`
- checkpoint evals: `runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71/output/checkpoint_eval_step000800.json`, `runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71/output/checkpoint_eval_step001200.json`
- official case-bank visualization: `runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71/output/case_bank/official/index.html`
- visualization hub: `runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71/visualizations/index.html`
- source snapshot: `runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71/source_snapshot.tar.gz`

## 7. Results

Run completed without abort on GPU1.

Training and runtime:

- final train CE: `0.98678`
- loop1 train loss: `1.17613`
- loop5 train loss: `0.98678`
- train time: `989.6s`
- max CUDA allocated: `48068.6 MB`
- max CUDA reserved: `50560.0 MB`
- final train feedback norm: `1.7137`
- final train corruption fraction: `0.3509`
- final corrupted-feedback confidence: `0.6923`

Checkpoint slope:

| step | CE | loop5 exact | loop5 blank acc | feedback norm |
|---:|---:|---:|---:|---:|
| 800 | `0.99035` | `0.01758` | `0.51084` | `1.1010` |
| 1200 | `0.93318` | `0.01855` | `0.52293` | `1.4944` |
| 1500 | `0.98678` | `0.02930` | `0.53081` | `1.7137` |

Final loop dynamics on official eval, `eval_n=2048`:

| loop | exact | blank acc | feedback in | feedback next |
|---:|---:|---:|---:|---:|
| 1 | `0.00635` | `0.45938` | `0.0000` | `1.7273` |
| 2 | `0.02930` | `0.51879` | `1.7273` | `1.7143` |
| 3 | `0.02930` | `0.52968` | `1.7143` | `1.7145` |
| 4 | `0.02930` | `0.53071` | `1.7145` | `1.7148` |
| 5 | `0.02930` | `0.53081` | `1.7148` | `1.7149` |

The feedback path is active, but it does not unlock late-loop exact. Exact jumps loop1->2 and then stays flat. Blank accuracy still creeps from loop3 to loop5 (`0.52968 -> 0.53081`), but full-board exact does not move.

Official case-bank, `eval_n=256`, blank counts `46-64`, mean `55.74`:

- final loop5 exact: `0.03516`
- final loop5 blank acc: `0.53073`
- selected visual cases: `3` solved-by-loop and `3` hard failures.

Case trajectories, loops `[1,2,3,5]`:

| case | holes | wrong cells |
|---|---:|---|
| `official_solved_by_loop_01_b0121` | 47 | `7,0,0,0` |
| `official_solved_by_loop_02_b0131` | 49 | `5,0,0,0` |
| `official_solved_by_loop_03_b0089` | 47 | `4,0,0,0` |
| `official_hard_failure_01_b0231` | 56 | `26,13,4,5` |
| `official_hard_failure_02_b0177` | 54 | `21,9,7,7` |
| `official_hard_failure_03_b0073` | 54 | `19,11,10,9` |

## 8. Conclusions

Decision: discard as a positive mechanism result.

What worked:

- The feedback/corruption mechanism is not dead code. `loop_feedback_next_norm` grew from `0.102` at step100 to `1.714` by step1500, with corruption fraction close to the requested `0.35`.
- Training remained stable at D192/L10/batch128 on GPU1 and kept the same rough exact plateau as the clean D192 baseline.
- The result gives a clean boundary: a generic prediction-feedback channel can be learned, but making it robust to corrupted previous probabilities is not by itself enough to create board-level correction.

What failed:

- Full-board exact did not exceed the existing D192/L10 plateau. Final loop5 exact is `0.029296875`, the same as the clean 1500-step baseline.
- Later loops still do not act like a sustained self-correction process. Loop2 reaches `0.02930` exact; loops 3/4/5 stay there.
- Hard cases do not show reliable loop3->loop5 repair. One hard case worsened `4 -> 5` wrong cells, one stayed `7 -> 7`, and one only nudged `10 -> 9`.

Mechanism lesson:

EqR-style "attractor" cannot be reduced to "feed previous predictions back with random corruption." That creates a trainable channel, but the channel learns a stable operating point rather than a better decision boundary. The next useful direction should change the state/update representation itself or introduce a more direct generic global-consistency target. Do not sweep corruption probability, mix, seed, or feedback scale.

Paper implication:

FutureSeed remains a strong opening mechanism for RWKV/GDN. This experiment narrows the missing piece: the current loop has enough signal to use feedback, but not enough structure to turn local blank accuracy into full-board exact. The honest story is now "FutureSeed opens cheap recurrent backbones; sustained exact reasoning requires better generic recurrent state dynamics."

## 9. Submission Record

None.
