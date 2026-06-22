# official-eqr-causal-fs-gate-20260622

## 1. Metainfo

- Plan ID: `P-EQR-006`
- Status: discarded as a positive result; completed as a mechanism gate
- Machine: AIStation GPU1 only
- Remote worktree: `/huyang2/double-loop/.worktrees/official-eqr-gate-787a612`
- Remote official EqR base: `/huyang2/double-loop/official_eqr_compare`
- Launcher source SHA at start: `118c245f804de717e99b95c4e116ae4851b80bfc`
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

- GPU: AIStation GPU1, A800 80GB, `CUDA_VISIBLE_DEVICES=0`
- Python/Torch: official EqR environment already present under
  `/huyang2/double-loop/official_eqr_compare/.venv`; model compiled with
  `torch.compile()`
- Optimizer: `adam_atan2`
- Attention backend: official EqR attention path patched to causal attention
- Notes: peak VRAM was not instrumented into the log; live monitor during the
  pair was about 4-5GB, so this run cannot support a precise VRAM claim.

## 5. Commands

Prepare causal clones:

```bash
CUDA_VISIBLE_DEVICES=0 OFFICIAL_EQR_BASE=/huyang2/double-loop/official_eqr_compare \
  /huyang2/double-loop/.worktrees/official-eqr-gate-787a612/scripts/official_eqr_compare/run_official_eqr_compare.sh prepare-causal
```

Train causal noFS:

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=256 TRAIN_EPOCHS_PER_ITER=256 GLOBAL_BATCH_SIZE=128 \
PATH_TOKEN_WEIGHT=8 RUN_NAME=official-eqr-causal-pathw8-e256-20260622T0640Z-118c245 \
  /huyang2/double-loop/.worktrees/official-eqr-gate-787a612/scripts/official_eqr_compare/run_official_eqr_compare.sh train-causal
```

Train causal FutureSeed:

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=256 TRAIN_EPOCHS_PER_ITER=256 GLOBAL_BATCH_SIZE=128 \
PATH_TOKEN_WEIGHT=8 FUTURE_SEED_SCALE=1 FUTURE_SEED_GATE_BIAS=-2 \
RUN_NAME=official-eqr-causal-fs-pathw8-e256-20260622T0645Z-118c245 \
  /huyang2/double-loop/.worktrees/official-eqr-gate-787a612/scripts/official_eqr_compare/run_official_eqr_compare.sh train-causal-futureseed
```

Visual/eval:

```bash
CUDA_VISIBLE_DEVICES=0 EQR_VISUAL_CONDITION=causal \
EQR_CHECKPOINT=/huyang2/double-loop/official_eqr_compare/outputs/causal/outputs/2vv420kh/2026-6-22/6-29-17/checkpoints/step_1792_2vv420kh.pth \
  /huyang2/double-loop/.worktrees/official-eqr-gate-787a612/scripts/official_eqr_compare/run_official_eqr_compare.sh eval-causal

CUDA_VISIBLE_DEVICES=0 EQR_VISUAL_CONDITION=causal_futureseed \
EQR_CHECKPOINT=/huyang2/double-loop/official_eqr_compare/outputs/causal-futureseed/outputs/vtp4wuzf/2026-6-22/6-34-9/checkpoints/step_1792_vtp4wuzf.pth \
  /huyang2/double-loop/.worktrees/official-eqr-gate-787a612/scripts/official_eqr_compare/run_official_eqr_compare.sh eval-causal-futureseed
```

## 6. Artifacts

- Local dashboard:
  `runs/official-eqr-causal-fs-gate-20260622/summary/index.html`
- Local hard-case visuals:
  `runs/official-eqr-causal-fs-gate-20260622/causal/index.html`
  and `runs/official-eqr-causal-fs-gate-20260622/causal_futureseed/index.html`
- Score/config:
  `runs/official-eqr-causal-fs-gate-20260622/score.json`,
  `runs/official-eqr-causal-fs-gate-20260622/config.json`
- Logs:
  `runs/official-eqr-causal-fs-gate-20260622/logs/causal-train.log`,
  `runs/official-eqr-causal-fs-gate-20260622/logs/causal-futureseed-train.log`
- Patches:
  `runs/official-eqr-causal-fs-gate-20260622/provenance/causal.patch`,
  `runs/official-eqr-causal-fs-gate-20260622/provenance/causal_futureseed.patch`

## 7. Results

Path-aware case visualization over 256 official Maze test cases:

| loop | base F1 | FS F1 | delta | base precision | FS precision | base recall | FS recall | base pred frac | FS pred frac | base FP | FS FP | base FN | FS FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.466220 | 0.466376 | +0.000157 | 0.304781 | 0.304899 | 0.999123 | 0.999252 | 0.434562 | 0.434453 | 271.91 | 271.79 | 0.11 | 0.09 |
| 4 | 0.468244 | 0.468188 | -0.000055 | 0.306427 | 0.306382 | 0.999966 | 0.999934 | 0.432578 | 0.432626 | 270.02 | 270.07 | 0.00 | 0.01 |
| 8 | 0.468211 | 0.468172 | -0.000038 | 0.306401 | 0.306367 | 0.999968 | 0.999935 | 0.432622 | 0.432643 | 270.06 | 270.09 | 0.00 | 0.01 |
| 16 | 0.468067 | 0.468157 | +0.000090 | 0.306275 | 0.306351 | 0.999970 | 1.000000 | 0.432791 | 0.432704 | 270.21 | 270.13 | 0.00 | 0.00 |

Training/eval diagnostics:

- Causal noFS params: `263,938`; train wall time: about `03:35`.
- Causal FutureSeed params: `296,708`; train wall time: about `03:24`.
- Visualization pass: noFS `10.09s` for 256 cases; FutureSeed `7.23s` for
  256 cases. This is not enough to claim inference speedup because the pair was
  not repeated or counterbalanced.
- Train step500 shows a transient FutureSeed optimization edge:
  accuracy `0.5557` vs `0.5202`, residual16 `173.684` vs `199.187`.
- By step1750 the train eval has converged to the same operating point:
  accuracy `0.699176` for both; residual16 `9.962` FS vs `9.836` noFS.
- Loop gain is tiny in both arms:
  noFS loop1 to loop16 F1 `+0.00185`, FutureSeed `+0.00178`.

## 8. Conclusions

FutureSeed did not meet the success gate on official EqR causalized Maze. The
final path F1 delta is only `+0.00009`, far below the required `+0.03`, and the
mask shape is unchanged: both conditions predict a very broad PATH mask with
recall essentially `1.0`, precision around `0.306`, and roughly `270` false
positive PATH cells per case.

The useful insight is narrower:

- FutureSeed can slightly accelerate early optimization after removing
  noncausal attention, but this advantage disappears by the final checkpoint.
- Causalizing EqR does not by itself make the Maze proxy expose the cheap
  bidirectional mechanism. The path-weighted objective still rewards broad
  coverage more than clean path selection.
- Loop is still not doing visible late correction here. Loop16 mostly lands on
  the same operating point as loop1, with only a tiny reduction in predicted
  PATH fraction.

Decision: do not claim FutureSeed beats official EqR on Maze from this result.
Do not sweep seed, gate bias, path weight, or causal attention details. The next
paper-useful experiment should either test a real EqR mixer-compression setting
where FutureSeed replaces compute, or move to a cleaner bidirectional diagnostic
where broad-mask shortcuts are not the dominant solution.

Kill criteria:

- official causal clones fail to prepare or import;
- causal noFS cannot train under the existing official dependency path;
- FutureSeed only improves token accuracy or recall by widening pred_path_frac
  and increasing FP;
- two consecutive failures reduce to weight/seed/temperature tweaking.

## 9. Submission Record

N/A.
