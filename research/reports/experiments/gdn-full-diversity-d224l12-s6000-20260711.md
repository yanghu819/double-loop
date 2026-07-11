# GDN Full-Diversity D224/L12 Scale Test

## 1. Metainfo

- Plan ID: `P-SCALE-029`
- Status: in-progress
- Planned: 2026-07-11 22:43 CST / 2026-07-11T14:43:00Z
- Launched: 2026-07-11 23:23 CST / 2026-07-11T15:23:21Z
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

## 8. Conclusions

Pending.

## 9. Submission Record

No tag unless the final mechanism/scaling result is strong and clean.
