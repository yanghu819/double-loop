# GDN Real-Backward Official Sudoku D192 Feedback-Attractor Probe

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-feedback-attractor-s1500-20260629`
- plan ID: `P-GDN-015`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: to be filled after launch commit
- local branch: `codex/gpu1-experiment-tracking`
- planned: 2026-06-29 07:08 UTC

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

To be filled after run completion.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

None.
