# Official FLA Linear Attention + FutureSeed Comparison

## 1. Metainfo

- Plan ID: `P-LA-002`
- Status: approved; implementation and strict provenance gate in progress
- Started: 2026-07-20 12:36 CST / 2026-07-20T04:36:00Z
- Machine: AIStation `GPU1` A800 only; GPU2 forbidden
- Branch: `codex/fla-gdn2-kda`
- Implementation base SHA: `648ac616964372385713e2eade15a899a1283326`
- Training SHA: pending clean commit
- Official FLA source SHA: `fe8fce9fc6984f22905f54cfa885dce1502baf26`
- Wheel SHA256: `65f57bf2aa937991fc497bd63f42883d263ead8646f21f2492735ccddd82d0eb`

## 2. Hypothesis

The earlier comparison did not isolate the Linear Attention recurrence. Its GDN
reference was the project's custom implementation with final-loop-only loss and
special residual-safe initialization, while KDA/GDN2 used official FLA layers
with every-loop loss. It therefore measured three implementation/training
changes at once.

This experiment asks one decision-changing question: under one pinned official
FLA implementation, one outer network, one recurrent-state shape, and one native
FutureSeed rule, does finer generic memory control improve difficult global
closure? KDA replaces GDN's per-head decay with per-key-channel decay. GDN2 adds
independent key-channel erase and value-channel write gates. The capacity
ordering is GDN2 superset KDA superset GDN, but finite-budget optimization need
not follow it.

Prediction: at least one of loss slope, 51-55-blank exact, or late-loop gain
should order `GDN2 >= KDA >= GDN`. A null result after all three train normally
would mean the extra state-update freedom is not useful at this task/compute
scale, not that the newer architecture is globally worse.

## 3. Configuration

- Arms: official FLA `GatedDeltaNet`, `KimiDeltaAttention`, and
  `GatedDeltaNet2`; all with native FutureSeed enabled.
- Official layer execution: call each class's complete `forward`; inject only
  the recurrent initial state through official FLA `Cache`.
- Shared model: D192/L10/H6/head-dim32, expand-v2, short-conv4, channel-mult4,
  loop5, all-loop CE, fixed unit-normalized FutureSeed scale1.
- Shared optimization: BF16, microbatch32, grad accumulation4, effective
  batch128, LR0.0015, AdamW weight decay0.001, seed52, 1500 optimizer steps.
- Shared data: `/huyang2/double-loop/data/sudoku-extreme-full`, train/test;
  curriculum `46-50:100,51-55:1400`.
- Evaluation: checkpoint 500/1000/1500 on 53 blanks; final official ranges
  46-50, 51-55, 56-64; loop1-5 and hard-case visualizations.
- Disabled: feature noise, task rules, repair, search, selector, best-of-K,
  architecture-specific LR/loss/seed tuning, CPU smoke, GPU2.

## 4. Environment

- Remote root: `/huyang2/double-loop`
- Python: `/opt/conda/bin/python`
- All caches: below `/huyang2/double-loop/.cache`
- Strict backend settings:
  - `CUDA_VISIBLE_DEVICES=0`
  - `FLA_DISABLE_BACKEND_DISPATCH=1`
  - `FLA_CONV_BACKEND=triton`
  - `FLA_STRICT_OFFICIAL=1`
- GPU/probe result: pending GPU1 queue

## 5. Commands

Strict provenance, reference, backward, initial-state-gradient, official-forward,
and FutureSeed-stack gate:

```bash
CUDA_VISIBLE_DEVICES=0 \
FLA_DISABLE_BACKEND_DISPATCH=1 FLA_CONV_BACKEND=triton \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
TRITON_CACHE_DIR=/huyang2/double-loop/.cache/triton \
TORCHINDUCTOR_CACHE_DIR=/huyang2/double-loop/.cache/torchinductor \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
TMPDIR=/huyang2/double-loop/.cache/tmp \
PYTHONPATH=/huyang2/double-loop/.cache/python-extra-pylib \
/opt/conda/bin/python \
  experiments/rwkv_fs_sudoku/check_fla_delta_backbones.py \
  --backbone all \
  --wheel /huyang2/double-loop/wheelhouse/flash_linear_attention-0.5.2-py3-none-any.whl \
  --out artifacts/fla-official-futureseed-strict-gate.json
```

Training, once per listed backbone and only after the gate passes:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_fla_official_futureseed_compare.sh \
  <fla_gdn|kda|gdn2> <resolved-run-name>
```

## 6. Artifacts

- Strict gate JSON/log: pending
- GDN run: pending
- KDA run: pending
- GDN2 run: pending
- Aggregate JSON/HTML/screenshots: pending
- Checkpoints stay below repository-local `models/` and are not committed.

## 7. Results

Pending.

Required table: parameters, training tokens, wall time, steady-state step time,
peak allocated/reserved VRAM, CE at 100/500/1000/1500, loop1-5 exact/blank,
official 46-50/51-55/56-64 exact/blank, and same-case wrong-cell evolution.

## 8. Conclusions

Pending. Kill conditions:

- any installed-source hash, source marker, official class path, expected chunk
  autograd node, CUDA initial-state gradient, or Triton backend mismatch;
- wrong GPU, CPU execution, NaN/OOM;
- step500 CE that has not materially fallen from initialization.

No seed/LR/loss rescue table will be run. If all three train normally but remain
closed, the decision is to stop architecture enumeration and return to the
proven data/compute scaling axis.

## 9. Submission Record

None.
