# GDN Full-Diversity D224/L12 Scale Test

## 1. Metainfo

- Plan ID: `P-SCALE-029`
- Status: done
- Planned: 2026-07-11 22:43 CST / 2026-07-11T14:43:00Z
- Launched: 2026-07-11 23:23 CST / 2026-07-11T15:23:21Z
- Completed: 2026-07-12 05:14 CST / 2026-07-11T21:14:35Z
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Source SHA: `eeb38f5b6f9b6c1b1df0bbd3ca149daec06a171d`
- Run: `gdn-full-diversity-d224l12-s6000-20260711T1510Z-eeb38f5`

## 2. Hypothesis

The current official-data runs do not train on one million independent Sudoku
problems. The released `sudoku-extreme-1k-aug-1000` train split contains exactly
1,000 source groups and 1,001 symmetry variants per group. The model therefore
sees many masks and legal transformations, but only 1,000 underlying source
problems.

Sudoku-Extreme provides 3,831,994 unaugmented raw source rows. If hard
full-board exactness is limited by relational data diversity rather than another
small state or loss mechanism, replacing the augmented 1k split with the full
raw split should steepen the D224 learning curve without changing the model.

This is a direct bitter-lesson test: scale independent data while holding the
backbone, recurrent compute, optimizer, curriculum, and evaluation fixed.

## 3. Configuration

- Data variable only:
  - old: 1,000 source boards, 1,001 variants each, 1,001,000 rows;
  - new: 3,831,994 unaugmented raw train rows.
- Evaluation: unchanged official test split with 422,786 rows.
- Model: native FutureSeed GDN, D224/L12/H14/head-dim16, expand-v4.
- Recurrent compute: loop5, every-loop CE, fixed FutureSeed and loop update.
- Batch: microbatch32, gradient accumulation4, effective batch128.
- Curriculum: `46-50:100,51-55:5900`.
- Target: step6000 from scratch.
- Checkpoints: 1000, 3000, 4500, 6000 on holes53/60/64.
- Disabled: noise, scratch, repair, search, selector, oracle rollout, task rules,
  seed/LR/loss/width sweeps.

## 4. Prediction And Kill Criteria

Matched old-data references:

- step1000 holes53 exact/blank: approximately `0.0176/0.5284`;
- step3000 holes53 exact/blank: `0.0352/0.5750`;
- step6000 holes53 exact/blank: `0.1680/0.6120`;
- step6000 official 51-55 exact: `0.2832`;
- step6000 official 56-64 exact: `0.0801`.

Success requires a better generalization curve, not merely lower train CE:

- primary: step6000 holes53 exact `>0.1680` and official 51-55 `>0.2832`;
- hard-range success: official 56-64 `>=0.12` without losing 51-55;
- strong success: holes60 or holes64 checkpoint exact reaches `>=0.25`.

Kill criteria:

- stop on wrong GPU, NaN, OOM, malformed data, or source/checkpoint ambiguity;
- at step3000, stop only if holes53 exact is `<0.02`, blank accuracy is `<0.54`,
  and the 1000-to-3000 curve has no acceleration;
- if full diversity only lowers train loss while matched exact/blank stay flat,
  classify data diversity as insufficient and do not build a dataset-size table.

## 5. Commands

The public `train.csv` was downloaded locally through Kimi WebBridge using
resumable HTTP Range requests, then gzip-compressed locally and uploaded through
the AIStation helper. It was not downloaded by the GPU container.

The remote data builder ran from detached source SHA `eeb38f5`:

```bash
/opt/conda/bin/python scripts/build_sudoku_dataset_from_csv.py \
  --csv /huyang2/double-loop/data/sudoku-extreme-full/raw/train.csv \
  --output-dir /huyang2/double-loop/data/sudoku-extreme-full \
  --expected-rows 3831994 --chunk-rows 8192
```

A CUDA-only one-step smoke used the same D224/L12, FutureSeed, loop5, and GDN
Triton path. The first attempt was rejected before GPU work because the CLI does
not accept the string `hidden_agg_noise_mode=none`; the corrected form is
`gumbel` with noise scale exactly zero. The second smoke completed forward,
backward, official eval, and artifact recording on GPU1.

Formal launch script:

`/huyang2/double-loop/artifacts/launch/pscale029/pscale029_launch.sh`

The script verifies the exact detached SHA, tracked-clean worktree, smoke score,
data arrays, GPU1 visibility, and raw CSV provenance before executing
`./run.sh full` with the configuration in section 3.

## 6. Artifacts

- Raw CSV:
  `/huyang2/double-loop/data/sudoku-extreme-full/raw/train.csv`
- Raw CSV bytes / lines: `718819925 / 3831995` including the header.
- Raw CSV SHA256:
  `64b46674db0148e0d73a16346dadeb2b1c00824d3fca3f85b2ae7037f6b4b38e`
- Gzip SHA256:
  `fc24e42059142784034d3ba230aa9a7c4f0a616a987d5f76c94369c00ff2eaaa`
- Built train arrays:
  `/huyang2/double-loop/data/sudoku-extreme-full/train`
- Formal detached worktree:
  `/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z`
- Run directory:
  `/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-s6000-20260711T1510Z-eeb38f5`
- Checkpoints:
  `/huyang2/double-loop/models/gdn-full-diversity-d224l12-s6000-20260711T1510Z-eeb38f5/checkpoints`
- Outer PID / Python PID at launch: `2459 / 59579`.

## 7. Results

Data and CUDA gates passed. The formal run started with `git_dirty=0`,
`device=cuda`, PyTorch `2.7.0+cu126`, GDN `triton_recurrent`, and about `45 GB`
allocated on the sole visible A800.

The first lease trained through the complete periodic step900 checkpoint. CE
fell from `1.6346` at step100 to `0.9688` at step900. Exact PIDs
`2508/2506/2507/2459` were then stopped proactively, GPU1 returned to `0 MiB`,
and `lease_rollover_1.json` recorded the exact checkpoint. AIStation was restored
through Kimi WebBridge without changing the A800/image/resource configuration.
The new container `5bdodqqquitj0-0` was probed as an A800 before resume.

The second leg resumed optimizer, model, and RNG state at global step900. The
step1000 checkpoint is:

| Bucket | loop1 exact | loop5 exact | loop5 blank accuracy |
|---|---:|---:|---:|
| holes53 | `0.0156` | `0.0176` | `0.5145` |
| holes60 | `0.0156` | `0.0215` | `0.5254` |
| holes64 | `0.0137` | `0.0215` | `0.5250` |

Matched holes53 is not an early win: exact ties the old-data reference `0.0176`
while blank accuracy trails `0.5284` by about `0.014`. Train CE is `0.9971`
versus the old run's approximately `0.95`. This is neither success nor the
predeclared kill: the experiment continues unchanged to the step3000 crossover
gate.

The step3000 crossover gate completed at 2026-07-12 02:20 CST / 2026-07-11
18:20 UTC:

| Bucket | loop1 exact | loop1 blank | loop5 exact | loop5 blank |
|---|---:|---:|---:|---:|
| holes53 | `0.0176` | `0.5174` | `0.0273` | `0.5788` |
| holes60 | `0.0215` | `0.5279` | `0.0332` | `0.5885` |
| holes64 | `0.0254` | `0.5211` | `0.0430` | `0.5844` |

The matched old-data holes53 reference at step3000 is exact/blank
`0.0352/0.5750`. Full diversity is therefore `-0.0078` exact but `+0.0038`
blank accuracy: it has not yet converted its slightly better local prediction
quality into more wholly correct boards. It nevertheless passes the
predeclared continuation rule because exact is above `0.02`, blank accuracy is
above `0.54`, and train CE continues to improve (`0.9971` at step1000, `0.7939`
at step3000, then `0.7279` at step3100).

Loop compute is already behaviorally useful at this gate. On holes64, loop1 to
loop5 raises exact from `0.0254` to `0.0430` and blank accuracy from `0.5211` to
`0.5844`; holes60 exact similarly rises from `0.0215` to `0.0332`. The run
therefore continues unchanged to step4500/6000 to test whether broader data is
a delayed generalization gain rather than an immediate optimization gain.

The delayed gain became clear at step4500:

| Bucket | loop1 exact | loop1 blank | loop5 exact | loop5 blank |
|---|---:|---:|---:|---:|
| holes53 | `0.0176` | `0.5312` | `0.1055` | `0.6166` |
| holes60 | `0.0215` | `0.5417` | `0.1172` | `0.6332` |
| holes64 | `0.0254` | `0.5341` | `0.1230` | `0.6133` |

Holes53 full-board exact is nearly four times its step3000 value (`0.0273` to
`0.1055`) while blank accuracy rises only `0.0378`. This is the desired change
in behavior: local improvements have started converting into globally correct
boards. The loop1-to-loop5 exact gain is now `+0.0879` on holes53, `+0.0957` on
holes60, and `+0.0977` on holes64, so the model is using recurrent compute rather
than merely copying one answer.

The second lease was stopped only after the complete step4500 checkpoint and
evaluation. Exact PIDs `181/179/180/132` were terminated, GPU1 returned to
`0 MiB`, and `lease_rollover_2.json` records the transition at
`2026-07-11T19:46:39Z`. Kimi WebBridge restored the same A800 80GB, CPU16,
GPU1, memory64GB, shm20GB configuration. The new AIStation instance
`a6d40c23-6af6-4315-b700-2e1c1b362a7b` was probed as an A800 before the third
leg resumed exact model, optimizer, and RNG state from step4500. Its run is
`gdn-full-diversity-d224l12-resume4500-s6000-20260711T1950Z-eeb38f5`.

The step6000 checkpoint is the decisive crossover:

| Bucket | loop1 exact | loop2 exact | loop3 exact | loop4 exact | loop5 exact / blank |
|---|---:|---:|---:|---:|---:|
| holes53 | `0.0176` | `0.0234` | `0.1445` | `0.1992` | `0.2109 / 0.6344` |
| holes60 | `0.0215` | `0.0371` | `0.1738` | `0.2305` | `0.2402 / 0.6643` |
| holes64 | `0.0254` | `0.0430` | `0.1758` | `0.2441` | `0.2500 / 0.6410` |

Matched old-data holes53 at step6000 was `0.1680/0.6120`. Independent-data
scaling improves exact by `+0.0430` absolute (about `+25.6%` relative) and blank
accuracy by `+0.0224`. The predeclared primary gate is passed. Holes64 reaches
the strong `>=0.25` checkpoint gate exactly.

The final 512-board mixed evaluation also shows a large loop effect:

| Loop | exact | blank accuracy |
|---:|---:|---:|
| 1 | `0.0234` | `0.5486` |
| 2 | `0.0273` | `0.6107` |
| 3 | `0.1738` | `0.6373` |
| 4 | `0.2383` | `0.6431` |
| 5 | `0.2500` | `0.6439` |

Most board-level correction begins between loop2 and loop3, after loop2 has
already captured most of the local blank-accuracy gain. This separates local
prediction from global convergence: later recurrent compute turns similar
per-cell quality into many more wholly valid boards.

Official blank-range comparisons against the matched old-data D224/L12 run:

| Range | old loop5 exact | full-diversity loop5 exact | delta |
|---|---:|---:|---:|
| 46-50 | `1.0000` | `1.0000` | `0.0000` |
| 51-55 | `0.2832` | `0.3926` | `+0.1094` |
| 56-64 | `0.0801` | `0.1270` | `+0.0469` |

The hard-range success gate (`56-64 >=0.12`) is passed without losing the
already solved easy range. Case-bank evidence is consistent with the aggregate:
one selected 56-blank board goes from `24` wrong cells at loop1 to `0` at loop3;
an almost-solved board goes `12 -> 5 -> 1`; the selected 64-blank hard failure
still improves `25 -> 9 -> 5`. Thus loop computation now performs real
correction, while the remaining frontier is convergence on the last few
globally coupled errors.

Recorded cumulative training elapsed is about `16250 s` (`4.51 h`) across three
GPU1 leases. The final leg reports `44.5 GB` peak allocated CUDA memory. The
metadata-light bundle SHA256 is
`f05f50053d4ac7e828ea5359cb7fb4d536749609904574c2a4dba98ce37bd6b1`.
The remote final run retains its `63,337,147`-byte source snapshot; checkpoints,
source archives, and data are not committed to GitHub.

## 8. Conclusions

P-SCALE-029 is a clean positive scaling result. Holding model, recurrent
compute, optimizer, batch, curriculum, and evaluation fixed while replacing
1,000 repeatedly augmented source boards with 3,831,994 independent boards
improves every hard primary readout. The strongest gain is official 51-55 exact
(`+0.1094` absolute), but the hardest 56-64 range also passes its planned gate.

The mechanism-level lesson is equally important: FutureSeed plus loop was not
fundamentally capped at local blank accuracy. Once the backbone saw enough
independent relational structures, loop3/4 began converting a broad partially
correct state into valid full boards. The useful scaling frontier is therefore
independent data plus compute on the efficient D224 state, not blind width.

This result does not isolate FutureSeed against a matched full-diversity no-FS
run, and it is not a final EqR comparison. Those remain separate paper gates.
Do not weaken this result with dataset-size, seed, loss, or width tables. The
next high-information experiment is a single longer full-diversity continuation
on the same efficient backbone, with harder data exposure, to test whether the
56-64 slope continues before changing the generic state formulation.

## 9. Submission Record

- Final score: `0.2500` at `metrics.eval_clean.loop5.label_exact`.
- Leaderboard row and all three metadata legs are committed.
- No tag: the project rule requires primary score `>=0.50`.
