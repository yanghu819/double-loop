# FLA GDN2/KDA FutureSeed Sudoku Gate

## 1. Metainfo

- Plan ID: `P-LA-001`
- Status: approved; implementation in progress
- Planned: 2026-07-19 11:00 CST / 2026-07-19T03:00:00Z
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/fla-gdn2-kda`
- Source SHA: pending clean implementation commit
- Runs: pending

## 2. Hypothesis

FutureSeed already supplies each deeper recurrent layer with the terminal state of
the preceding layer. The unresolved question is whether the state update itself
is too coarse for difficult global closure.

The current GDN v1 state has one decay and one erase/write strength per head.
KDA uses a separate decay for each key channel. GDN2 additionally separates
channel-wise erase on the key axis from channel-wise write on the value axis.
These are generic learned memory operations, not Sudoku rules.

If state-edit granularity is the bottleneck, matched training should show a
mechanistic ordering: GDN2 opens or closes hard boards earlier than KDA, and KDA
earlier than GDN v1. If both new recurrences only improve cell accuracy without
improving full-board exact, the missing ingredient is not a richer delta-rule
state and this branch should stop.

## 3. Configuration

- Upstream implementation: official FLA commit
  `fe8fce9fc6984f22905f54cfa885dce1502baf26`.
- Wheel: locally built `flash_linear_attention-0.5.2-py3-none-any.whl`, SHA256
  `65f57bf2aa937991fc497bd63f42883d263ead8646f21f2492735ccddd82d0eb`.
- Task/data: official EqR Sudoku arrays,
  `sudoku-extreme-1k-aug-1000` train/test.
- Model: D128/L6/H8/head-dim16, channel multiplier4, expand-v1.
- Recurrent compute: loop4, every-loop CE, fixed native FutureSeed scale1.
- Training: 600 steps, batch128, BF16, LR `0.0015`, weight decay `0.001`.
- Arms: one `BACKBONE=gdn2`, one `BACKBONE=kda`.
- Matched reference: P-GDN-005 GDN v1 FutureSeed, exact `0.0107`, blank accuracy
  `0.4925`, CE `1.0667`, training `212.06s`, peak allocation `15380.5MB`.
- Disabled: noise, scratch, repair, search, selector, task rules, seed/LR/loss
  sweeps, and CPU smoke.

## 4. Environment

- GPU row: `GPU1`; `CUDA_VISIBLE_DEVICES=0` only.
- Expected GPU: NVIDIA A800-SXM4-80GB.
- Remote root: `/huyang2/double-loop`.
- Python: `/opt/conda/bin/python`, PyTorch `2.7.0+cu126`.
- All caches, wheel installs, checkpoints, logs, and runs remain below the
  repository root.

## 5. Commands

CUDA integration gate, including an actual three-layer FutureSeed stack at the
training head dimension, to run from a clean detached worktree:

```bash
CUDA_VISIBLE_DEVICES=0 \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
TRITON_CACHE_DIR=/huyang2/double-loop/.cache/triton \
TORCHINDUCTOR_CACHE_DIR=/huyang2/double-loop/.cache/torchinductor \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
TMPDIR=/huyang2/double-loop/.cache/tmp \
PYTHONPATH=/huyang2/double-loop/.cache/python-extra-pylib \
  /opt/conda/bin/python \
  experiments/rwkv_fs_sudoku/check_fla_delta_backbones.py \
  --backbone gdn2 \
  --out artifacts/fla-delta-kernel-check.json
```

The first cold-cache combined invocation was stopped by exact process group
`4157` at `20m28s`, as required by the integration-gate kill criterion. It had
not reported a numerical mismatch; it was still generating FLA autotune
kernels and had produced 3,848 persistent Triton cache files. The checker now
accepts `--backbone gdn2|kda`, logs each sub-check as it starts and passes, and
writes partial JSON after every passed sub-check. The split reruns reuse the
same repository-local cache and make a compile stall distinguishable from a
forward/backward/state mismatch.

The split GDN2 run passed its strict recurrence reference and real adapter
checks. Kernel versus Torch maximum errors were `5.99e-5` for output,
`2.76e-4` for terminal state, and `2.34e-4` over all gradients including the
initial state. The BF16 adapter had a nonzero initial-state gradient norm
`0.00735`; full versus split-sequence error was `0.00781` for output and
`0.00279` for terminal state; measured forward plus backward was `7.90ms` at
the test shape.

The initial three-layer check then exposed a checker bug rather than a kernel
failure: it treated every `grad=None` as non-finite, even though
`blocks.0.future_seed_logit` is structurally unused because layer zero has no
preceding terminal state to seed it. The corrected check permits only that
named missing gradient, reports any other missing gradient separately from
NaN/Inf, and supports `--check futureseed_stack` so the already-passed kernel
tests do not need to be rerun.

After both backbones passed, the first GDN2 training launch
`fla-gdn2-futureseed-sudoku-gate-20260719T041850Z-986e725` stopped before its
first optimizer step. P-GDN-005 was recorded before official-data blank-range
filtering existed, so its nominal default `2-4` was ignored and it actually
sampled uniformly from the full 46-64-blank train split. The current loader
correctly rejected the impossible range. The matched launcher now sets
`HOLES_MIN=46` and `HOLES_MAX=64`; because that is the complete train split,
the seeded index sampling is semantically identical to the historical
baseline. No curriculum or difficulty reweighting was introduced.

Training command template:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
OFFICIAL_SUDOKU_TRAIN_SPLIT=train OFFICIAL_SUDOKU_EVAL_SPLIT=test \
SUDOKU_SIZE=9 HOLE_PATTERN=random EVAL_HOLES=16 EVAL_HOLES_LIST=16 \
BACKBONE=<gdn2-or-kda> GDN_MODE=chunk GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=1.0 \
D_MODEL=128 LAYERS=6 HEADS=8 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 MAX_LOOPS=4 \
LOOP_LOSS=all FULL_STEPS=600 FULL_BATCH=128 FULL_EVAL_N=1024 \
FULL_ROLLOUT_KS= FULL_LOG_EVERY=100 FORWARD_DTYPE=bfloat16 \
LR=0.0015 WEIGHT_DECAY=0.001 BLANK_LOSS_WEIGHT=8 FUTURE_SEED_SCALE=1 \
RUN_NAME=<resolved-name> ./run.sh full
```

## 6. Artifacts

Pending. Each completed or aborted run must archive config, score, logs, source
SHA/snapshot, CUDA check, visual HTML, and hardest-case JSON.

## 7. Results

Pending.

## 8. Conclusions

Decision rules:

- stop before training if either FLA kernel disagrees with the CUDA Torch
  reference in output, terminal state, backward, or initial-state gradient;
- kill an arm at step300 if CE remains above `1.5` with no blank-accuracy
  opening, or if wall time per step exceeds matched GDN by more than `2x`;
- scale only a clear winner: exact at least `+0.01` or blank accuracy at least
  `+0.03` over GDN v1 with less than `50%` wall overhead;
- equivalent quality with at least `20%` lower wall time or VRAM also counts;
- do not interpret a blank-accuracy-only gain as global-reasoning success.

## 9. Submission Record

None.
