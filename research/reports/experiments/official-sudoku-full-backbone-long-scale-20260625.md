# Official Sudoku Full Backbone Long Scale

## 1. Metainfo

- Plan ID: `P-SUDOKU-002`
- Status: done
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-06-25 14:58:07 +0800`
- Execution window: `2026-06-25T07:19Z` to `2026-06-25T08:16Z`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`

## 2. Hypothesis

The 600-step official-data gate showed that the full standalone
FutureSeed-RWKV-loop backbone opens while the matched no-FS backbone stays on a
high-loss plateau. The high-value question is whether this opening compounds
with more training, or whether it saturates at a tiny nonzero exact rate.

Prediction:

- If FutureSeed is a real scalable optimization mechanism for this backbone,
  the 1500-step FS arm should improve clearly beyond the 600-step exact
  `0.0137`, ideally `>=0.05`, with positive loop5-loop1 gain.
- If step count is not the bottleneck, FS will keep CE near `1.0` and exact
  near the previous tiny value; then the next axis should be state dynamics or
  model/data scale, not more same-config epochs.
- A matched no-FS arm is worth running only if the FS arm opens enough to need
  a fair delta.

## 3. Configuration

- Data: official EqR Sudoku Extreme arrays,
  `sudoku-extreme-1k-aug-1000`
- Dataloader: standalone FutureSeed runner official `.npy` path
- Architecture: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- Train: `FULL_STEPS=1500`, `FULL_BATCH=128`, official train split
- Eval: official test split fixed subset, `FULL_EVAL_N=2048`
- First arm: `FUTURE_SEED_SCALE=1`
- Conditional matched arm: `FUTURE_SEED_SCALE=0` only if FS opens
- Rollout metrics: explicitly disabled with `FULL_ROLLOUT_KS=`
- Forbidden: right-to-left scan, solver, repair, selector, oracle rollout,
  Sudoku-specific postprocessing, seed sweep

## 4. Environment

- CUDA device: GPU1 only via `CUDA_VISIBLE_DEVICES=0`
- AIStation row: `GPU1`
- Work platform: `d9cfb2d6-2033-4063-81f0-e9cb21e52b4e`
- Pod: `chvjskqscre30-0`
- GPU visible in container: `NVIDIA A800-SXM4-80GB`
- Worktree: `/huyang2/double-loop/.worktrees/official-sudoku-long-bd338ad`
- Git SHA: `bd338ad36f1f5b7fc59f25b7629a435930f86844`
- Python: `/opt/conda/bin/python`
- Torch: `2.7.0+cu126`
- RWKV kernel: CUDA statepassing
- Caches/artifacts/runs/models under `/huyang2/double-loop`
- Both launches recorded `git_dirty=0`.

## 5. Commands

No CPU smoke. Syntax/static validation only before push:

```bash
bash -n run.sh
python -m py_compile experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py
```

Remote launch shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
PIP_CACHE_DIR=/huyang2/double-loop/.cache/pip \
HF_HOME=/huyang2/double-loop/.cache/huggingface \
TORCH_HOME=/huyang2/double-loop/.cache/torch \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
PATH=/huyang2/double-loop/.cache/bin:$PATH \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 D_MODEL=192 LAYERS=10 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 FULL_STEPS=1500 FULL_BATCH=128 FULL_EVAL_N=2048 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 RWKV_KERNEL=statepassing FORWARD_DTYPE=bfloat16 \
FUTURE_SEED_SCALE=1 RUN_NAME=official-sudoku-fullbackbone-long-fs-<timestamp>-<sha> \
./run.sh full
```

Actual runs:

- FS:
  `official-sudoku-fullbackbone-long-fs-20260625T071931Z-bd338ad`
- no-FS:
  `official-sudoku-fullbackbone-long-nofs-20260625T074809Z-bd338ad`

The no-FS arm was launched after the FS arm exceeded the previous 600-step
score and showed useful loop gain. The detached worktree `leaderboard.csv` was
restored before no-FS launch because `run.sh full` still mutates it through
`record_experiment.py`; both actual run configs were clean.

Kill criteria:

- If step1000 exact remains `<=0.02` and CE remains `>1.0`, stop after
  archiving and do not launch no-FS.
- If step100 takes more than 20 minutes, stop by exact PID and relaunch smaller
  `FULL_BATCH=80`.
- If source is dirty or `FULL_ROLLOUT_KS=` does not disable rollout metrics,
  stop before using the result.

## 6. Artifacts

- FS run:
  `/huyang2/double-loop/.worktrees/official-sudoku-long-bd338ad/runs/official-sudoku-fullbackbone-long-fs-20260625T071931Z-bd338ad`
- no-FS run:
  `/huyang2/double-loop/.worktrees/official-sudoku-long-bd338ad/runs/official-sudoku-fullbackbone-long-nofs-20260625T074809Z-bd338ad`
- FS launch:
  `/huyang2/double-loop/artifacts/launch/official-sudoku-fullbackbone-long-fs-20260625T071931Z-bd338ad`
- no-FS launch:
  `/huyang2/double-loop/artifacts/launch/official-sudoku-fullbackbone-long-nofs-20260625T074809Z-bd338ad`
- FS visualization:
  `/huyang2/double-loop/.worktrees/official-sudoku-long-bd338ad/runs/official-sudoku-fullbackbone-long-fs-20260625T071931Z-bd338ad/output/futureseed_loop_case_seed52.html`
- no-FS visualization:
  `/huyang2/double-loop/.worktrees/official-sudoku-long-bd338ad/runs/official-sudoku-fullbackbone-long-nofs-20260625T074809Z-bd338ad/output/futureseed_loop_case_seed52.html`
- Remote cleanup: no training process remained, GPU usage returned to `0 MiB /
  0%`, source status was clean after restoring detached `leaderboard.csv`, and
  `/tmp/aistation-ssh-password` was removed.

## 7. Results

Training CE:

| step | FS CE | no-FS CE | FS - no-FS |
|---:|---:|---:|---:|
| 100 | 1.8832 | 1.8809 | +0.0023 |
| 200 | 1.5627 | 1.6850 | -0.1223 |
| 300 | 1.2112 | 1.6607 | -0.4495 |
| 400 | 1.1420 | 1.6470 | -0.5050 |
| 500 | 1.0891 | 1.6516 | -0.5625 |
| 600 | 1.0632 | 1.6486 | -0.5854 |
| 700 | 1.0667 | 1.6421 | -0.5754 |
| 800 | 1.0044 | 1.6388 | -0.6344 |
| 900 | 1.0386 | 1.6475 | -0.6089 |
| 1000 | 1.0350 | 1.6301 | -0.5951 |
| 1100 | 1.0137 | 1.6722 | -0.6585 |
| 1200 | 0.9456 | 1.6219 | -0.6763 |
| 1300 | 1.0043 | 1.6277 | -0.6234 |
| 1400 | 1.0320 | 1.6563 | -0.6243 |
| 1500 | 1.0000 | 1.6312 | -0.6312 |

Final official test metrics, `eval_n=2048`:

| arm | loop | exact | blank_acc | early | late |
|---|---:|---:|---:|---:|---:|
| no-FS | 1 | 0.0000 | 0.2671 | 0.2031 | 0.3340 |
| no-FS | 5 | 0.0000 | 0.2697 | 0.2043 | 0.3371 |
| FS | 1 | 0.0015 | 0.4473 | 0.4428 | 0.4539 |
| FS | 5 | 0.0293 | 0.5291 | 0.5283 | 0.5313 |

Loop readout:

- no-FS loop gain: exact `+0.0000`, blank_acc `+0.0026`.
- FS loop gain: exact `+0.0278`, blank_acc `+0.0818`.
- FS loop5 exact improved from the 600-step gate `0.0137` to `0.0293`.
- no-FS remained on the high-loss plateau through step1500 and never reached
  nonzero exact.
- `FULL_ROLLOUT_KS=` worked as intended: no stochastic rollout metrics were
  produced, and no selector/repair/oracle rollout was used.

## 8. Conclusions

Mixed-positive, high-information result.

FutureSeed is a real optimization/sample-efficiency mechanism for the full
standalone backbone on official EqR Sudoku arrays. Under matched 1500-step
budget, no-FS stays stuck at CE `1.6312`, exact `0`, and almost no loop gain.
FutureSeed reaches CE `1.0000`, exact `0.0293`, blank accuracy `0.5291`, and
large loop refinement from loop1 to loop5.

But same-config epoch scaling is not enough. FS roughly doubles the 600-step
exact score, yet it remains far below a strong Sudoku result and below the
pre-run ideal `>=0.05`. The next useful experiment should not be another
same-size epoch extension. The higher-value axes are more capacity/data,
harder curriculum design, or a simple generic FutureSeed/loop state dynamics
change that helps later loops turn better optimization into stronger exact.

## 9. Submission Record

Not applicable.
