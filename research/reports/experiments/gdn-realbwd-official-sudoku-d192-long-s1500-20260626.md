# GDN Real-Backward Official Sudoku D192 Long Scaling

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-long-s1500-20260626`
- plan ID: `P-GDN-007`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: `fce43c9755baca7db873ef0516f0f428b21aae9b`
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-26 UTC

## 2. Hypothesis

P-GDN-006 showed a clean mechanism split: D192/L10 no-FS did not open, while D192/L10 FutureSeed opened at 600 steps. The next high-ROI question is not another no-FS control. It is whether the opened GDN+FutureSeed model has a useful scaling slope when trained longer.

Prediction: if GDN+FutureSeed is a viable backbone rather than a tiny opening trick, 1500 steps should materially improve exact/blank accuracy over the 600-step result (`loop5 exact 0.0176`, `blank_acc 0.5021`) and should preserve a positive loop1-to-loop5 gain.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=5`
- batch: `128`
- steps: `1500`
- eval_n: `2048`
- dtype: `bfloat16`
- only arm: `FUTURE_SEED_SCALE=1`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep, no no-FS repeat.

## 4. Environment

- AIStation row: GPU1 only
- host: `6c5froee38e7t-0`
- GPU: NVIDIA A800-SXM4-80GB
- remote worktree: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-b471906-20260626`
- Python: `/opt/conda/bin/python`
- torch: `2.7.0+cu126`
- observed peak memory: about `49.6GB`
- run metadata reports `git_dirty=false`

## 5. Commands

Launch at source SHA `fce43c9755baca7db873ef0516f0f428b21aae9b`:

```bash
SOURCE_SHA=fce43c9755baca7db873ef0516f0f428b21aae9b \
RUN_STAMP=20260626T0910Z \
WT_OVERRIDE=/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-b471906-20260626 \
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_long_s1500_20260626.sh
```

## 6. Artifacts

- run: `runs/gdn-d192-official-sudoku-fs-long-s1500-20260626T0910Z-fce43c9`
- visual: `runs/gdn-d192-official-sudoku-fs-long-s1500-20260626T0910Z-fce43c9/visualizations/index.html`
- result JSON: `runs/gdn-d192-official-sudoku-fs-long-s1500-20260626T0910Z-fce43c9/output/futureseed_loop_seed52.json`
- score: `runs/gdn-d192-official-sudoku-fs-long-s1500-20260626T0910Z-fce43c9/score.json`
- remote log: `/huyang2/double-loop/artifacts/logs/gdn-d192-official-sudoku-fs-long-s1500-20260626T0910Z-fce43c9.log`

## 7. Results

Training curve:

| step | CE | loop1 CE | loop5 CE |
|---:|---:|---:|---:|
| 100 | 1.8048 | 1.8050 | 1.8048 |
| 200 | 1.3207 | 1.3282 | 1.3207 |
| 300 | 1.1287 | 1.2053 | 1.1287 |
| 400 | 1.0853 | 1.2112 | 1.0853 |
| 500 | 1.0488 | 1.1436 | 1.0488 |
| 600 | 1.0479 | 1.1842 | 1.0479 |
| 700 | 1.0430 | 1.1799 | 1.0430 |
| 800 | 1.0103 | 1.1666 | 1.0103 |
| 900 | 1.0172 | 1.2363 | 1.0172 |
| 1000 | 1.0150 | 1.2440 | 1.0150 |
| 1100 | 0.9848 | 1.1944 | 0.9848 |
| 1200 | 0.9462 | 1.1453 | 0.9462 |
| 1300 | 0.9881 | 1.1646 | 0.9881 |
| 1400 | 1.0296 | 1.2144 | 1.0296 |
| 1500 | 0.9875 | 1.2053 | 0.9875 |

Full-board eval, official test split, `eval_n=2048`:

| loop | exact | blank_acc | early | late | early-late gap |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0054 | 0.4547 | 0.4417 | 0.4571 | -0.0154 |
| 2 | 0.0293 | 0.5177 | 0.5153 | 0.5230 | -0.0077 |
| 3 | 0.0293 | 0.5323 | 0.5312 | 0.5358 | -0.0046 |
| 4 | 0.0293 | 0.5334 | 0.5324 | 0.5362 | -0.0038 |
| 5 | 0.0293 | 0.5337 | 0.5330 | 0.5367 | -0.0037 |

Comparisons:

- vs GDN+FS 600-step: loop5 exact `0.0176 -> 0.0293`, blank_acc `0.5021 -> 0.5337`.
- loop refinement strengthened: loop1->loop5 exact gain `+0.0239`; blank_acc gain `+0.0790`.
- vs historical RWKV D192/L10 1500-step: exact is tied at about `0.0293`; blank_acc is slightly better (`0.5337` vs about `0.5291`).
- FS state is active: `fs_gate=0.488`, `fs_state_norm=7.810`.

## 8. Conclusions

Decision: mark P-GDN-007 done and keep GDN+FutureSeed as a serious scaling branch.

This run answers the intended question. GDN+FutureSeed does not merely produce a tiny 600-step opening; with longer training it improves to the same exact level as the historical RWKV+FutureSeed 1500-step run and slightly higher blank accuracy. This supports the idea that FutureSeed is a portable recurrent-state mechanism rather than RWKV-specific plumbing.

The loop signal is also stronger than the 600-step run. Exact accuracy jumps from `0.0054` at loop1 to `0.0293` by loop2 and stays there, while blank accuracy keeps improving through loop5. That means later loops are still doing useful refinement, though exact saturates early by loop2 for this checkpoint.

Limits:

- Absolute score is still low; no `exp/score-*` tag.
- Training CE is noisy after step1200 (`0.9462 -> 0.9875` by step1500), so a blind 3000-step continuation is not automatically high ROI.
- This remains official 9x9 Sudoku; it strengthens the backbone/scaling story but not Maze/path reasoning.

Next high-ROI decision:

- Prefer one D256/L10 or D256/L12 memory/throughput gate over immediate 3000-step continuation. This run already shows longer D192 helps and reaches RWKV 1500-level; the next better-lesson question is whether more capacity gives a steeper scaling slope.

## 9. Submission Record

None.
