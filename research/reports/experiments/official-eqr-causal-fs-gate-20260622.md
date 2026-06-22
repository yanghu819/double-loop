# official-eqr-causal-fs-gate-20260622

## 1. Metainfo

- Plan ID: `P-EQR-006`
- Status: in-progress
- Machine: AIStation GPU1 only
- Remote worktree: `/huyang2/double-loop/.worktrees/official-eqr-gate-787a612`
- Remote official EqR base: `/huyang2/double-loop/official_eqr_compare`
- Launcher source SHA at start: `787a612feaacd06af60127293fcd3fe98e4455a4`
- Date: 2026-06-22 Asia/Shanghai

## 2. Hypothesis

FutureSeed should be tested as cheap bidirectional information, not as an
unattributable add-on after EqR's full noncausal mixer. If the claim is real,
then causalizing the EqR token mixer should expose a gap where FutureSeed helps
recover future-side information under the same official codebase, data, budget,
and path-aware evaluation.

## 3. Configuration

- Official EqR upstream SHA: `aba94e9cde0f273ce644db5261cd6915ba6561f0`
- Dataset: official `maze-30x30-unique-1k`
- Objective: existing path-aware EqR objective with `path_token_weight=8`,
  used only because previous official-token-loss runs produced near-zero PATH
  predictions and are not an informative path-recovery benchmark.
- Baseline boundary: existing full noncausal EqR pathw8 e256 run, final loop16
  path F1 around `0.4682`, but as a broad mask.
- New matched pair:
  - `causal`: EqR attention uses `causal=true`; no FutureSeed.
  - `causal-futureseed`: same causal attention plus FutureSeed hooks.
- Budget: `EPOCHS=256`, `GLOBAL_BATCH_SIZE=128`, `eval loops=1,4,8,16`.
- Forbidden: selector, search, repair, oracle rollout, maze-specific rules,
  CPU smoke, GPU2.

## 4. Environment

To be filled from remote launch:

- GPU:
- Python:
- Torch:
- AdamATan2 backend:
- Attention backend:

## 5. Commands

Prepare causal clones:

```bash
CUDA_VISIBLE_DEVICES=0 OFFICIAL_EQR_BASE=/huyang2/double-loop/official_eqr_compare \
  /huyang2/double-loop/.worktrees/official-eqr-gate-787a612/scripts/official_eqr_compare/run_official_eqr_compare.sh prepare-causal
```

Train causal noFS:

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=256 TRAIN_EPOCHS_PER_ITER=256 GLOBAL_BATCH_SIZE=128 \
PATH_TOKEN_WEIGHT=8 RUN_NAME=official-eqr-causal-pathw8-e256-20260622TBD \
  /huyang2/double-loop/.worktrees/official-eqr-gate-787a612/scripts/official_eqr_compare/run_official_eqr_compare.sh train-causal
```

Train causal FutureSeed:

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=256 TRAIN_EPOCHS_PER_ITER=256 GLOBAL_BATCH_SIZE=128 \
PATH_TOKEN_WEIGHT=8 FUTURE_SEED_SCALE=1 FUTURE_SEED_GATE_BIAS=-2 \
RUN_NAME=official-eqr-causal-fs-pathw8-e256-20260622TBD \
  /huyang2/double-loop/.worktrees/official-eqr-gate-787a612/scripts/official_eqr_compare/run_official_eqr_compare.sh train-causal-futureseed
```

Visual/eval commands will be recorded after checkpoint paths are known.

## 6. Artifacts

Pending.

## 7. Results

Pending.

Required metrics:

- exact
- token accuracy
- path F1
- path precision
- path recall
- pred_path_frac
- FP / FN
- train wall time
- eval throughput
- VRAM
- params
- loop1 versus loop16 deltas

## 8. Conclusions

Pending.

Kill criteria:

- official causal clones fail to prepare or import;
- causal noFS cannot train under the existing official dependency path;
- FutureSeed only improves token accuracy or recall by widening pred_path_frac
  and increasing FP;
- two consecutive failures reduce to weight/seed/temperature tweaking.

## 9. Submission Record

N/A.
