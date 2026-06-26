# GDN Real-Backward Official Sudoku D192 Scale

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-scale-20260626`
- plan ID: `P-GDN-006`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: `b47190644a97ace3d51283923d8a8806437ce628`
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-26 UTC

## 2. Hypothesis

P-GDN-005 showed that native FutureSeed transfers to Gated DeltaNet: D128/L6 no-FS plateaued, while D128/L6 FS opened with loop4 exact `0.0107`.

The next question is scale: if GDN is a viable FutureSeed backbone, D192/L10/loop5 should approach the earlier RWKV full-backbone 600-step official Sudoku opening (`loop5 exact 0.0137`, blank_acc `0.4972`) instead of staying at the D128 gate level.

This is not a table. It is staged: run FS first; only run matched no-FS if FS opens enough to make the control worth the GPU time.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=5`
- batch target: `128`, fallback `80` only for OOM or step100 speed kill
- steps: `600`
- eval_n: `2048`
- dtype: `bfloat16`
- first arm: `FUTURE_SEED_SCALE=1`
- conditional arm: `FUTURE_SEED_SCALE=0` only if FS opens

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep.

## 4. Environment

- AIStation row: GPU1 only
- host: `6c5froee38e7t-0`
- GPU: NVIDIA A800-SXM4-80GB
- remote worktree: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-b471906-20260626`
- Python: `/opt/conda/bin/python`
- torch: `2.7.0+cu126`
- observed peak memory: about `49.6GB`
- both completed runs report `git_dirty=0`

## 5. Commands

Clean FS launch:

```bash
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_scale_fs_clean_b471906_20260626.sh
```

Matched no-FS launch:

```bash
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_scale_nofs_reuse_fswt_b471906_20260626.sh
```

The first attempted FS launcher wrote `.codex-run-gdn-d192-fs.sh` inside the detached worktree and was stopped before use as evidence. It is archived as:

```text
runs/gdn-d192-official-sudoku-fs-s600-20260626T0748-b471906/abort.json
```

## 6. Artifacts

- FS run: `runs/gdn-d192-official-sudoku-fs-clean-s600-20260626T0825-b471906`
- no-FS run: `runs/gdn-d192-official-sudoku-nofs-reusefswt-s600-20260626T0855-b471906`
- FS visual: `runs/gdn-d192-official-sudoku-fs-clean-s600-20260626T0825-b471906/visualizations/index.html`
- no-FS visual: `runs/gdn-d192-official-sudoku-nofs-reusefswt-s600-20260626T0855-b471906/visualizations/index.html`
- remote FS log: `/huyang2/double-loop/artifacts/logs/gdn_d192_scale_fs_clean_b471906_20260626.log`
- remote no-FS log: `/huyang2/double-loop/artifacts/logs/gdn_d192_scale_nofs_reuse_fswt_b471906_20260626.log`

## 7. Results

Training curve:

| arm | step100 CE | step200 CE | step300 CE | step400 CE | step500 CE | step600 CE |
|---|---:|---:|---:|---:|---:|---:|
| no-FS D192/L10 | 1.8453 | 1.7587 | 1.6663 | 1.6429 | 1.6423 | 1.6369 |
| FS D192/L10 | 1.8048 | 1.3207 | 1.1287 | 1.0853 | 1.0488 | 1.0479 |

Full-board eval, official test split, `eval_n=2048`:

| arm | loop1 exact | loop5 exact | loop gain | loop1 blank_acc | loop5 blank_acc | fs_gate | fs_state_norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-FS D192/L10 | 0.0000 | 0.0000 | 0.0000 | 0.2663 | 0.2674 | 0.000 | 0.000 |
| FS D192/L10 | 0.0034 | 0.0176 | +0.0142 | 0.4603 | 0.5021 | 0.491 | 7.862 |

Comparisons:

- FS vs no-FS at loop5: exact `+0.0176`, blank_acc `+0.2347`, step600 CE `-0.5890`.
- FS D192/L10 beats the earlier D128/L6 FS gate: loop exact `0.0176` vs `0.0107`, blank_acc `0.5021` vs `0.4925`.
- FS D192/L10 also exceeds the historical RWKV D192/L10 600-step opening on exact (`0.0176` vs `0.0137`) and is close on blank_acc (`0.5021` vs about `0.4972`).

## 8. Conclusions

Decision: mark P-GDN-006 done and keep GDN as a live FutureSeed backbone.

The useful result is not just that D192/L10 is larger. The matched no-FS control stayed at the same plateau as smaller no-FS GDN (`CE≈1.64`, exact `0`, blank_acc `~0.267`). The FS arm crossed into the opened regime (`CE≈1.05`, nonzero exact, blank_acc `~0.50`) under the same model size, data, batch, loop count, steps, and eval.

This supports a clean mechanism claim: native FutureSeed is acting as a generic recurrent-state bridge for GDN, not an RWKV-specific trick. It also gives a modest but real loop signal under FS: loop1 to loop5 exact improves by `+0.0142` and blank_acc by `+0.0418`; without FS, loop is almost neutral.

Limits:

- The absolute score is still low and not tag-worthy; no `exp/score-*` tag.
- This is still official 9x9 Sudoku, not Maze or broader benchmark evidence.
- The no-FS setup tried a separate worktree first, but `git worktree add` stalled; the final no-FS run reused the clean FS worktree after restoring `leaderboard.csv`, and run metadata confirms `git_dirty=0`.
- The no-FS `source.patch` contains a `runs/visualization_index.html` artifact diff from worktree reuse. It is not a source/model code diff.

Next high-ROI step:

- Do not run more tiny gates. Either train FS D192/L10 longer (`1500-3000` steps) to see if it approaches RWKV long-run behavior, or move one controlled axis to D256/L10 if memory allows. The decision question should be whether GDN+FutureSeed has better scaling slope than RWKV+FutureSeed, not whether another small no-FS remains dead.

## 9. Submission Record

None.
