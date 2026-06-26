# GDN Real-Backward Official Sudoku Scale Gate

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-scale-20260626`
- plan ID: `P-GDN-005`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: `be2815bd27ece12df237ed01456e651d1dac5212`
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-26 UTC

## 2. Hypothesis

The real Triton backward removes the GDN training bottleneck. The next question is no longer whether the kernel runs; it is whether native FutureSeed transfers beyond RWKV.

If FutureSeed is a generic terminal-state source of cheap future context, then a larger Gated DeltaNet backbone should show better opening/sample-efficiency with FutureSeed than without it on the same official Sudoku data. If the gain is RWKV-specific, matched no-FS and FS GDN should move together.

This is one matched scale gate, not a seed sweep.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0` to isolate the local recurrent state path
- size: `D_MODEL=128`, `LAYERS=6`, `HEADS=8`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=4`
- batch target: `128`; fallback `96` or `64` only for OOM/speed kill
- steps: `600`
- dtype: `bfloat16`
- arms:
  - no-FS: `FUTURE_SEED_SCALE=0`
  - FS: `FUTURE_SEED_SCALE=1`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep.

## 4. Environment

- GPU row: GPU1
- GPU: NVIDIA A800-SXM4-80GB
- torch: `2.7.0+cu126`
- CUDA visible devices: `0`
- worktree no-FS: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-scale-be2815b-20260626`
- worktree FS: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-scale-fs-clean-be2815b-20260626`

## 5. Commands

No-FS launch:

```bash
/huyang2/double-loop/artifacts/launch/gdn_realbwd_official_sudoku_scale_be2815b_20260626.sh
```

The original paired script completed no-FS, then FS became dirty because `record_experiment.py` updated `leaderboard.csv` in the same worktree after no-FS. That dirty FS partial was stopped and marked with `abort.json`. The clean FS arm was relaunched from a separate detached worktree:

```bash
/huyang2/double-loop/artifacts/launch/gdn_realbwd_official_sudoku_scale_fs_clean_be2815b_20260626.sh
```

Both final arms used:

```bash
SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
PYTHON_BIN=/opt/conda/bin/python CUDA_VISIBLE_DEVICES=0 \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
OFFICIAL_SUDOKU_TRAIN_SPLIT=train OFFICIAL_SUDOKU_EVAL_SPLIT=test \
SUDOKU_SIZE=9 HOLE_PATTERN=random EVAL_HOLES=16 EVAL_HOLES_LIST=16 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=1.0 \
D_MODEL=128 LAYERS=6 HEADS=8 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 MAX_LOOPS=4 \
FULL_STEPS=600 FULL_BATCH=128 FULL_EVAL_N=1024 FULL_ROLLOUT_KS= FULL_LOG_EVERY=100 \
FORWARD_DTYPE=bfloat16 LR=0.0015 WEIGHT_DECAY=0.001 BLANK_LOSS_WEIGHT=8 \
FUTURE_SEED_SCALE=<0-or-1> ./run.sh full
```

## 6. Artifacts

- no-FS run: `runs/gdn-realbwd-official-sudoku-scale-nofs-20260626T0055-be2815b`
- FS clean run: `runs/gdn-realbwd-official-sudoku-scale-fs-clean-20260626T0105-be2815b`
- stopped dirty partial: `runs/aborted-dirty-fs-partial/abort.json`
- launch scripts:
  - `artifacts/launch/gdn_realbwd_official_sudoku_scale_be2815b_20260626.sh`
  - `artifacts/launch/gdn_realbwd_official_sudoku_scale_fs_clean_be2815b_20260626.sh`
- logs:
  - `artifacts/logs/gdn_realbwd_official_sudoku_scale_be2815b_20260626.log`
  - `artifacts/logs/gdn_realbwd_official_sudoku_scale_fs_clean_be2815b_20260626.log`

## 7. Results

Training:

| arm | clean SHA | train sec | train CE | loop1 loss | loop4 loss | CUDA alloc MB |
|---|---:|---:|---:|---:|---:|---:|
| no-FS | yes | `200.80` | `1.6355` | `1.6393` | `1.6355` | `15194.9` |
| FS | yes | `212.06` | `1.0667` | `1.2253` | `1.0667` | `15380.5` |

Training curve:

| step | no-FS CE | FS CE | FS delta |
|---:|---:|---:|---:|
| 100 | `1.8137` | `1.7918` | `-0.0219` |
| 200 | `1.6591` | `1.3550` | `-0.3041` |
| 300 | `1.6523` | `1.1813` | `-0.4710` |
| 400 | `1.6400` | `1.1387` | `-0.5013` |
| 500 | `1.6418` | `1.0861` | `-0.5557` |
| 600 | `1.6355` | `1.0667` | `-0.5688` |

Official test eval, `eval_n=1024`:

| arm | loop | exact | blank_acc | valid | fs_gate | fs_state_norm |
|---|---:|---:|---:|---:|---:|---:|
| no-FS | 1 | `0.0000` | `0.2666` | `0.0000` | `0.000` | `0.000` |
| no-FS | 2 | `0.0000` | `0.2685` | `0.0000` | `0.000` | `0.000` |
| no-FS | 3 | `0.0000` | `0.2683` | `0.0000` | `0.000` | `0.000` |
| no-FS | 4 | `0.0000` | `0.2686` | `0.0000` | `0.000` | `0.000` |
| FS | 1 | `0.0010` | `0.4444` | `0.0010` | `0.476` | `7.617` |
| FS | 2 | `0.0078` | `0.4910` | `0.0078` | `0.476` | `7.617` |
| FS | 3 | `0.0107` | `0.4927` | `0.0107` | `0.476` | `7.617` |
| FS | 4 | `0.0107` | `0.4925` | `0.0107` | `0.476` | `7.617` |

Primary deltas at loop4:

- exact: `+0.0107`
- blank_acc: `+0.2239`
- train CE: `-0.5688`
- training time overhead: `+5.6%`
- peak memory overhead: `+185.6 MB`
- FS internal state active: `fs_gate=0.476`, `fs_state_norm=7.617`

## 8. Conclusions

This is a positive mechanism result.

What it proves:

- FutureSeed is not RWKV-only. With the same GDN backbone and real Triton backward, no-FS stays on a high-CE plateau while native-FS opens strongly.
- The gain is much larger than the overhead: FS adds about `5.6%` train time and about `186 MB` peak memory, while cutting train CE by `0.569` and raising official-test blank accuracy by `0.224`.
- Loop has useful refinement only when FutureSeed opens the state. no-FS loop1 to loop4 stays exact `0`; FS goes from loop1 exact `0.0010` to loop3/4 exact `0.0107`.

What it does not prove:

- It is not a final high-score Sudoku result. FS exact `0.0107` is still low.
- It does not yet show GDN beats RWKV FutureSeed or EqR. This run is smaller than the D192/L10 RWKV official run and only 600 steps.
- It does not validate the stopped dirty FS partial; that partial was aborted for provenance and replaced by the clean FS run.

Decision:

- Keep GDN as a live FutureSeed extension.
- Next high-ROI step is scale, not more gates: either D192/L10 matched no-FS/FS for 600-1500 steps, or D128/L6 longer hard-budget if we want a cheaper throughput curve.
- Do not add Sudoku repair, selector, search, or rule priors. The right next axis is model/data/compute and simple generic FutureSeed state dynamics.

## 9. Submission Record

None.
