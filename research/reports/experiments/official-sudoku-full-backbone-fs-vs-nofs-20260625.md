# official-sudoku-full-backbone-fs-vs-nofs-20260625

## 1. Metainfo

- Plan ID: `P-SUDOKU-001`
- Status: done
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-06-25 10:49:59 +0800`
- Clean execution window: `2026-06-25T06:25:33Z` to `2026-06-25T06:45Z`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`

## 2. Hypothesis

The previous native FutureSeed EqR gate only tested a 2-layer RWKV mixer
transplant inside official EqR. It should not be used to judge the original
standalone FutureSeed-RWKV-loop method.

Mechanism hypothesis: the complete recurrent backbone needs depth, looped
latent refinement, and official data training together. If the original method
is real rather than a random-hole artifact, it should begin to learn official
Sudoku-Extreme when trained directly on the official arrays. FutureSeed should
improve sample-efficiency or loop refinement relative to the same RWKV-loop
backbone with `future_seed_scale=0`.

Prediction: FutureSeed will show higher official-test blank accuracy and/or
earlier nonzero exact than no-FS under the same budget. If both arms remain
near-zero, the next question becomes clean scaling on official Sudoku, not more
2-layer EqR mixer patching.

## 3. Configuration

- Dataset: official EqR `sudoku-extreme-1k-aug-1000`
- Data mode: map official token `1` to this runner's blank id, and `2..10` to
  digit ids `0..8`
- Model: standalone `FutureSeedLoopSudoku`
- Size: 9x9, 3x3 boxes
- Architecture: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- Train: `FULL_STEPS=600`, `FULL_BATCH=128`, official train split
- Eval: official test split fixed subset, `FULL_EVAL_N=2048`
- Arms:
  - no-FS: `FUTURE_SEED_SCALE=0`
  - FS: `FUTURE_SEED_SCALE=1`
- Forbidden: right-to-left scan, solver, repair, selector, oracle rollout,
  Sudoku-specific postprocessing, seed sweep
- Note: this was reduced from the planned 1500-step gate to a 600-step paired
  viability gate because GPU1 lease time made a clean paired 1500+1500 run less
  useful than finishing both arms. This is still a valid mechanism gate because
  both arms used the same budget.
- Caveat: `FULL_ROLLOUT_KS=` currently falls back to `run.sh` defaults
  (`1,4,8,16`) rather than disabling rollout metrics. Rollouts were evaluation
  diagnostics only; they did not drive training, selection, or repair.

## 4. Environment

- CUDA device: GPU1 only via `CUDA_VISIBLE_DEVICES=0`
- GPU: NVIDIA A100-SXM4-80GB
- Python: `/opt/conda/bin/python`
- Torch: `2.7.0+cu126`
- RWKV kernel: CUDA statepassing
- Worktree: `/huyang2/double-loop/.worktrees/official-sudoku-full-f1f5905`
- Git SHA: `f1f59057f55f768964324f792d1896facc79c0fd`
- Clean runs had `git_dirty=false`
- Caches/artifacts/runs/models remained under `/huyang2/double-loop`
- Infrastructure fix: `/opt/conda/bin/ninja --version` exits `245`, so the
  project-local `/huyang2/double-loop/.cache/bin/ninja` compatibility wrapper
  had to be placed first in `PATH`; `TORCH_EXTENSIONS_DIR` was set to
  `/huyang2/double-loop/.cache/torch_extensions`; `ulimit -c 0` prevented core
  dumps from dirtying the worktree.

## 5. Commands

Static checks:

```bash
python -m py_compile experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py
bash -n run.sh
```

Clean command shape. The no-FS arm ran first; the auto-started FS arm was
stopped because no-FS had updated `leaderboard.csv`, making FS dirty. The
leaderboard mutation was restored in the detached worktree, then FS was
relaunched clean with `UPDATE_LEADERBOARD=0`.

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
PIP_CACHE_DIR=/huyang2/double-loop/.cache/pip \
HF_HOME=/huyang2/double-loop/.cache/huggingface \
TORCH_HOME=/huyang2/double-loop/.cache/torch \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
PATH=/huyang2/double-loop/.cache/bin:$PATH \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 D_MODEL=192 LAYERS=10 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 FULL_STEPS=600 FULL_BATCH=128 FULL_EVAL_N=2048 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 RWKV_KERNEL=statepassing FORWARD_DTYPE=bfloat16 \
FUTURE_SEED_SCALE=<0-or-1> RUN_NAME=<run> ./run.sh full
```

Kill criteria:

- If the first 100 steps take more than 20 minutes, stop by exact PID and
  relaunch a smaller viability probe with `FULL_BATCH=80`.
- If both arms are still high-loss with official eval blank accuracy near random
  after 1500 steps, do not sweep seeds; classify the next axis as official
  Sudoku scaling.
- If no-FS does not learn at all but FS does, continue the full-backbone
  direction and scale one axis only.

## 6. Artifacts

- no-FS run:
  `/huyang2/double-loop/.worktrees/official-sudoku-full-f1f5905/runs/official-sudoku-fullbackbone-short-nofs-20260625T062528Z-f1f5905`
- FS run:
  `/huyang2/double-loop/.worktrees/official-sudoku-full-f1f5905/runs/official-sudoku-fullbackbone-short-fs-clean-20260625T063618Z-f1f5905`
- no-FS visualization:
  `/huyang2/double-loop/.worktrees/official-sudoku-full-f1f5905/runs/official-sudoku-fullbackbone-short-nofs-20260625T062528Z-f1f5905/output/futureseed_loop_case_seed52.html`
- FS visualization:
  `/huyang2/double-loop/.worktrees/official-sudoku-full-f1f5905/runs/official-sudoku-fullbackbone-short-fs-clean-20260625T063618Z-f1f5905/output/futureseed_loop_case_seed52.html`
- Failed/dirty launch attempts were marked with `abort.json` and not used for
  conclusions.

## 7. Results

Training CE:

| step | no-FS CE | FS CE | FS - no-FS |
|---:|---:|---:|---:|
| 100 | 1.8809 | 1.8832 | +0.0023 |
| 200 | 1.6850 | 1.5627 | -0.1223 |
| 300 | 1.6607 | 1.2112 | -0.4495 |
| 400 | 1.6470 | 1.1420 | -0.5050 |
| 500 | 1.6516 | 1.0891 | -0.5625 |
| 600 | 1.6486 | 1.0632 | -0.5854 |

Final official test metrics, `eval_n=2048`:

| arm | loop | exact | blank_acc | early | late |
|---|---:|---:|---:|---:|---:|
| no-FS | 1 | 0.0000 | 0.2586 | 0.2010 | 0.3198 |
| no-FS | 5 | 0.0000 | 0.2618 | 0.2025 | 0.3230 |
| FS | 1 | 0.0015 | 0.4586 | 0.4593 | 0.4627 |
| FS | 5 | 0.0137 | 0.4972 | 0.4959 | 0.4989 |

Loop readout:

- no-FS loop gain: exact `+0.0000`, blank_acc `+0.0032`.
- FS loop gain: exact `+0.0122`, blank_acc `+0.0386`.
- FS loop-last train CE at step600 was `1.0632` vs loop1 CE `1.1687`,
  showing later loops were useful under FutureSeed.
- FutureSeed diagnostics at loop5: `fs_gate=0.522`, `fs_update=1.000`,
  `fs_state_norm=8.353`.
- Rollout diagnostics did not add oracle gap: K1/K4/K8/K16 all matched the
  same loop5 exact `0.0137`. This was diagnostic only, not selector use.

## 8. Conclusions

Positive mechanism result.

The failed official EqR RWKV mixer transplant should not be treated as a
verdict on the original method. When the full standalone FutureSeed-RWKV-loop
backbone is trained directly on official EqR Sudoku arrays, FutureSeed changes
the training dynamics decisively. The no-FS control hits a CE plateau around
`1.65` and exact remains `0`; the FS arm drops to CE `1.06`, doubles blank
accuracy relative to no-FS, reaches nonzero exact, and shows real loop
refinement from loop1 to loop5.

This is not yet a strong Sudoku score. It is a strong opening/sample-efficiency
signal. The next high-ROI step is a longer clean FS/no-FS official-data scale
test with rollout metrics explicitly disabled and no leaderboard mutation
between paired arms. Do not return to EqR mixer patch seed/gate sweeps.

## 9. Submission Record

Not applicable.
