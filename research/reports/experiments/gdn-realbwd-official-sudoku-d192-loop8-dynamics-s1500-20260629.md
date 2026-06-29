# GDN Real-Backward Official Sudoku D192 Loop8 Dynamics Probe

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-loop8-dynamics-s1500-20260629`
- plan ID: `P-GDN-013`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: `83b6bff27fe2bb520b73cf25d9514de0078a990e`
- local branch: `codex/gpu1-experiment-tracking`
- launched: 2026-06-29 04:16 UTC
- recorded: 2026-06-29 04:41 UTC

## 2. Hypothesis

The recent D192/L10 GDN+FutureSeed results all have the same shape: soft metrics improve, exact stalls around `0.03`, and indirect objectives do not open the plateau. The next question is whether the loop itself still has useful late-loop computation, or whether the state has already settled by loop3/5.

Mechanism question: if we train/evaluate with loop8, do loops 5->8 keep correcting official Sudoku boards, or do they copy the same operating point?

Prediction: if loop depth is a real remaining scaling axis, loop8 should improve exact beyond loop5 and hardest-case visualizations should show wrong hidden cells decreasing from loop5 to loop8. If loop3/5/8 are nearly identical, the bottleneck is not simply more recurrent compute; it is state capacity or state update dynamics.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- training blank-count range: `45-64`, full official train/test range
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=8`
- batch: `96`
- steps: `1500`
- eval_n: `2048`
- checkpoint eval: `800,1200`, no train checkpoint save
- case bank: official eval fixed batch, `CASE_BANK_N=3`, `CASE_BANK_EVAL_N=256`, loops `1,2,3,5,8`
- dtype: `bfloat16`
- optimizer: `LR=0.0015`, `WEIGHT_DECAY=0.001`
- blank CE weight: `8`
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_UPDATE=fixed`
- no exact-margin or extra loss

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule in training, no seed sweep, no loss table.

## 4. Environment

- host: AIStation `GPU1`
- GPU target: A100 80GB, `CUDA_VISIBLE_DEVICES=0`
- remote run worktree: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-loop8-dynamics-s1500-<sha>-20260629`
- git dirty at run start: must be `false` after detached checkout

## 5. Commands

Launch from GitHub-truth SHA:

```bash
SOURCE_SHA=83b6bff27fe2bb520b73cf25d9514de0078a990e \
RUN_STAMP=20260629T0416Z \
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_loop8_dynamics_s1500_20260629.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exact PID and record infrastructure failure.
- Step100 wall time above 10 minutes: loop8 is too slow for this diagnostic.
- If checkpoint800 has CE above `1.5` and loop8 exact is still zero with blank_acc below `0.45`, stop as optimization failure.
- If checkpoint1200 shows loop8 equals loop5 and runtime is already high, finish only if final eval is near; otherwise archive as loop-depth boundary.

Success criteria:

- Strong: loop8 exact `>=0.04`.
- Useful: loop5->loop8 exact gain `>=0.005`, or official hardest cases show loop5->loop8 wrong_count reduction without new wrong-cell inflation.
- Negative: loop3/5/8 exact and wrong_count are nearly identical.

## 6. Artifacts

- run root: `runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff`
- config: `runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff/config.json`
- score: `runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff/score.json`
- log: `runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff/logs/run.log`
- result JSON/MD/HTML: `runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff/output/futureseed_loop_seed52.{json,md,html}`
- checkpoint evals: `runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff/output/checkpoint_eval_step000800.json`, `runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff/output/checkpoint_eval_step001200.json`
- official case-bank visualization: `runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff/output/case_bank/official/index.html`
- visualization hub: `runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff/visualizations/index.html`
- source snapshot: `runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff/source_snapshot.tar.gz`

## 7. Results

Run completed without abort on GPU1.

Training and runtime:

- final train CE: `1.01335`
- loop1 train loss: `1.13477`
- loop8 train loss: `1.01335`
- train time: `1409.1s`
- max CUDA allocated: `57779 MB`
- max CUDA reserved: `61044 MB`

Checkpoint slope:

| step | CE | loop8 exact | loop8 blank acc |
|---:|---:|---:|---:|
| 800 | `1.04360` | `0.01416` | `0.50523` |
| 1200 | `1.01517` | `0.01807` | `0.51424` |
| 1500 | `1.01335` | `0.02930` | `0.52262` |

Final loop dynamics on official eval, `eval_n=2048`:

| loop | exact | blank acc |
|---:|---:|---:|
| 1 | `0.00830` | `0.46507` |
| 2 | `0.02881` | `0.50477` |
| 3 | `0.02930` | `0.52152` |
| 4 | `0.02930` | `0.52233` |
| 5 | `0.02930` | `0.52253` |
| 6 | `0.02930` | `0.52265` |
| 7 | `0.02930` | `0.52265` |
| 8 | `0.02930` | `0.52262` |

The aggregate gain is front-loaded. Loop1->loop2 moves exact by `+0.02051`; loop2->loop3 moves exact by only `+0.00049`; loop3->loop8 exact gain is exactly `0.00000`. Blank accuracy keeps creeping from loop3 to loop8 (`0.52152 -> 0.52262`), but this does not convert into more solved boards.

Official hardest-case bank, `eval_n=256`, blank counts `46-64`, mean `55.74`:

- final loop8 exact: `0.03516`
- final loop8 blank acc: `0.52225`
- selected visual cases: `3` solved-by-loop and `3` hard failures.

Case trajectories, loops `[1,2,3,5,8]`:

| case | holes | wrong cells | changed cells | conflict units |
|---|---:|---|---|---|
| `official_solved_by_loop_01_b0089` | 47 | `6,1,0,0,0` | `0,5,1,0,0` | `13,3,0,0,0` |
| `official_solved_by_loop_02_b0121` | 47 | `5,1,0,0,0` | `0,4,1,0,0` | `8,3,0,0,0` |
| `official_solved_by_loop_03_b0131` | 49 | `4,1,0,0,0` | `0,5,1,0,0` | `12,3,0,0,0` |
| `official_hard_failure_01_b0038` | 56 | `26,11,5,5,5` | `0,27,8,0,0` | `25,18,8,8,8` |
| `official_hard_failure_02_b0229` | 55 | `24,11,7,8,8` | `0,28,12,1,0` | `24,21,16,17,17` |
| `official_hard_failure_03_b0073` | 54 | `17,14,10,8,8` | `0,15,6,2,0` | `23,21,18,16,16` |

## 8. Conclusions

This is a negative loop-depth boundary, not a reason to sweep larger `MAX_LOOPS`.

What worked:

- FutureSeed+GDN still performs real early iterative refinement. The solved visual cases show a clear pattern: loop1 has several wrong/conflicting hidden cells, loop2 removes most of them, and loop3 solves the board.
- The official case-bank visualization now runs on actual official eval batches rather than synthetic holes, so the visual evidence matches the benchmark distribution.

What failed:

- Later loops do not keep repairing hard boards. In hard failures, loop5->loop8 changes `0/0/0` cells in two cases and only freezes the remaining wrong cells; aggregate exact is flat from loop3 through loop8.
- The checkpoint slope is mostly optimization/soft-accuracy slope, not exact-opening slope. Step800->1500 improves loop8 exact `0.01416 -> 0.02930`, but final loop depth itself does not unlock additional exact after loop3.

Decision:

- Do not run loop12/16/24 table-filling on this same D192/L10 fixed-state setup.
- Do not run 6000-step same-config training unless an intermediate slope shows exact opening; prior 1500/3000 plus this loop8 run already shows the same plateau.
- Next highest-ROI direction is state/capacity that changes the recurrent state itself, not another indirect loss: for example a single GDN value/state expansion probe (`GDN_EXPAND_V > 1`) or one simple generic loop-state update that can keep alternative state information alive after loop3.

Paper implication:

FutureSeed remains a strong opening mechanism versus no-FS controls, but the current loop mechanism is not yet strong evidence for sustained late-loop reasoning. The clean claim today is "FutureSeed helps the recurrent backbone open hard Sudoku under official data"; the stronger claim "more loop keeps correcting hard boards" is not supported by this run.

## 9. Submission Record

None.
