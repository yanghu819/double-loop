# official-eqr-h96-fs-gate-20260622

## 1. Metainfo

- Plan ID: `P-EQR-007`
- Status: discarded
- Machine: AIStation GPU1 only
- Remote worktree: `/huyang2/double-loop/.worktrees/official-eqr-gate-787a612`
- Remote official EqR base: `/huyang2/double-loop/official_eqr_compare`
- Launcher code commit: `72dedd97cc0937b6e22904d7d646f916a8150925`
- Remote checkout SHA: `72dedd97cc0937b6e22904d7d646f916a8150925`
- Date: 2026-06-22 Asia/Shanghai

## 2. Hypothesis

FutureSeed should be tested as a cheap bidirectional/context mechanism under a
real capacity or compute bottleneck, not as a small add-on to a full EqR mixer.
If it has paper-level value inside the official EqR codebase, then a compressed
EqR model should benefit more from FutureSeed than the full D128 model did.

This run compresses the official Maze EqR train config from `hidden_size=128` to
`hidden_size=96` while keeping `num_heads=8`. That reduces attention, MLP, state,
embedding, and readout width in a generic way; it does not add task-specific
rules, repair, search, selector, or oracle decoding.

## 3. Configuration

- Official EqR upstream SHA: `aba94e9cde0f273ce644db5261cd6915ba6561f0`
- Dataset: official `maze-30x30-unique-1k`
- Objective: path-token-weighted official EqR token objective,
  `PATH_TOKEN_WEIGHT=8`
- Pair:
  - compressed base: `arch.hidden_size=96 arch.num_heads=8`
  - compressed FutureSeed: same plus `arch.future_seed_scale=1`,
    `arch.future_seed_gate_bias=-2`
- Budget: `EPOCHS=256`, `TRAIN_EPOCHS_PER_ITER=256`,
  `GLOBAL_BATCH_SIZE=128`, visual eval loops `1,4,8,16`, max cases `256`
- Existing boundary references:
  - full official EqR D128 pathw8 loop16 F1 about `0.46824`
  - full official EqR D128+FutureSeed pathw8 loop16 F1 about `0.46835`
  - causalized EqR D128+FutureSeed delta only `+0.00009`

## 4. Environment

- AIStation row: GPU1 only
- Host: `c6kh0j2k6jo50-0`
- GPU: NVIDIA A800-SXM4-80GB
- Remote root: `/huyang2/double-loop`
- Official EqR base: `/huyang2/double-loop/official_eqr_compare`
- Python env: `/huyang2/double-loop/official_eqr_compare/.venv`

## 5. Commands

Train compressed base:

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=256 TRAIN_EPOCHS_PER_ITER=256 GLOBAL_BATCH_SIZE=128 \
PATH_TOKEN_WEIGHT=8 EQR_EXTRA_OVERRIDES='arch.hidden_size=96 arch.num_heads=8' \
RUN_NAME=official-eqr-h96-base-pathw8-e256-20260622T0710Z-72dedd9 \
  /huyang2/double-loop/.worktrees/official-eqr-h96-gate/scripts/official_eqr_compare/run_official_eqr_compare.sh train-base
```

Train compressed FutureSeed:

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=256 TRAIN_EPOCHS_PER_ITER=256 GLOBAL_BATCH_SIZE=128 \
PATH_TOKEN_WEIGHT=8 EQR_EXTRA_OVERRIDES='arch.hidden_size=96 arch.num_heads=8' \
FUTURE_SEED_SCALE=1 FUTURE_SEED_GATE_BIAS=-2 \
RUN_NAME=official-eqr-h96-fs-pathw8-e256-20260622T0716Z-72dedd9 \
  /huyang2/double-loop/.worktrees/official-eqr-h96-gate/scripts/official_eqr_compare/run_official_eqr_compare.sh train-futureseed
```

Visual/eval commands:

```bash
ART=/huyang2/double-loop/official_eqr_compare/artifacts/official-eqr-h96-fs-gate-20260622
/huyang2/double-loop/official_eqr_compare/.venv/bin/python scripts/official_eqr_compare/visualize_official_eqr_cases.py \
  --repo /huyang2/double-loop/official_eqr_compare/eqr-clean \
  --checkpoint /huyang2/double-loop/official_eqr_compare/outputs/base/outputs/oz2y29i2/2026-6-22/7-52-54/checkpoints/step_1792_oz2y29i2.pth \
  --out-dir "$ART/base" --run-name official-eqr-h96-base-pathw8-e256-cases-20260622 \
  --condition h96_base --dataset-data-path /huyang2/double-loop/official_eqr_compare/eqr-clean/data/maze-30x30-unique-1k \
  --batch-size 16 --max-cases 256 --num-vis-cases 8 --steps 1,4,8,16
/huyang2/double-loop/official_eqr_compare/.venv/bin/python scripts/official_eqr_compare/visualize_official_eqr_cases.py \
  --repo /huyang2/double-loop/official_eqr_compare/eqr-futureseed \
  --checkpoint /huyang2/double-loop/official_eqr_compare/outputs/futureseed/outputs/8fqokij1/2026-6-22/7-58-17/checkpoints/step_1792_8fqokij1.pth \
  --out-dir "$ART/futureseed" --run-name official-eqr-h96-futureseed-pathw8-e256-cases-20260622 \
  --condition h96_futureseed --dataset-data-path /huyang2/double-loop/official_eqr_compare/eqr-clean/data/maze-30x30-unique-1k \
  --batch-size 16 --max-cases 256 --num-vis-cases 8 --steps 1,4,8,16
```

## 6. Artifacts

- Remote artifact root: `/huyang2/double-loop/official_eqr_compare/artifacts/official-eqr-h96-fs-gate-20260622`
- Local archive: `runs/official-eqr-h96-fs-gate-20260622`
- Summary: `runs/official-eqr-h96-fs-gate-20260622/summary/summary.json`
- Summary HTML: `runs/official-eqr-h96-fs-gate-20260622/summary/index.html`
- Base visuals: `runs/official-eqr-h96-fs-gate-20260622/base/index.html`
- FutureSeed visuals: `runs/official-eqr-h96-fs-gate-20260622/futureseed/index.html`
- Base checkpoint: `/huyang2/double-loop/official_eqr_compare/outputs/base/outputs/oz2y29i2/2026-6-22/7-52-54/checkpoints/step_1792_oz2y29i2.pth`
- FutureSeed checkpoint: `/huyang2/double-loop/official_eqr_compare/outputs/futureseed/outputs/8fqokij1/2026-6-22/7-58-17/checkpoints/step_1792_8fqokij1.pth`

## 7. Results

Final path-aware loop16 on 256 official test cases:

| condition | params | train wall | path F1 | precision | recall | pred PATH frac | FP/case | FN/case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H96 base | 111,938 | ~3m08s | 0.467337 | 0.305645 | 1.000000 | 0.433689 | 271.02 | 0.00 |
| H96 FutureSeed | 130,372 | ~3m20s | 0.467160 | 0.305495 | 1.000000 | 0.433906 | 271.21 | 0.00 |

Delta: FutureSeed path F1 `-0.000178`; pred PATH frac `+0.000217`; FP `+0.20`.

## 8. Conclusions

Discard as positive Maze evidence. Compressing the official EqR hidden size to
H96 does not expose a FutureSeed advantage. Both arms reproduce the same
high-recall broad-mask operating point: recall is `1.0`, precision remains about
`0.306`, and false positives remain about `271` cells per case.

This completes the EqR mixer-compression gate. Do not sweep hidden size, gate
bias, seed, or path weight. The next Maze work must change the generic recurrent
state/training dynamics so later loops can leave the broad-mask attractor.

Predeclared kill criteria checked:

- Any run leaves GPU1 or uses CPU smoke.
- Official EqR data/config path differs between arms.
- FutureSeed only changes token accuracy while path F1/FP/FN stay unchanged.
- FutureSeed gain comes from larger pred_path_frac / broader mask.
- Both compressed arms reproduce the same broad-mask operating point with
  `|delta path F1| < 0.01`; then stop Maze positive-evidence experiments and
  keep Maze only as failure analysis.

## 9. Submission Record

N/A.
