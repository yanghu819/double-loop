# Loss Alignment: EqR vs FutureSeed/GDN

## 1. Metainfo

- run_name: `loss-alignment-eqr-vs-futureseed-20260630`
- plan_id: `P-DIAG-001`
- machine: local metadata analysis, no GPU training launched
- local repo: `/Users/torusmini/Documents/double-loop-gpu1/.codex-transfer/localrepo`
- generated_at_utc: `2026-06-30T03:58:39+00:00`
- source_parent_sha: `6846154e23df4b227fa44b55a4d95eb6219dc3fd`

## 2. Hypothesis

The next useful step is not another Sudoku loss/noise sweep. The useful question is narrower:

Does the gap to official EqR come from a missing EqR-style loss/noise ingredient, or from a more basic FutureSeed/GDN state-dynamics bottleneck?

Prediction: if EqR's loss/noise is the missing piece, then aligned evidence should show our clean path missing a specific training pressure that explains the plateau. If CE and blank accuracy improve while full-board exact stays flat across clean training, noise, curriculum, and weak objectives, then the bottleneck is not a small loss tweak; it is conversion from per-cell confidence to board-level consistency.

## 3. Configuration

No training configuration was launched. The diagnostic script reads existing artifacts from:

- official EqR reproduction records:
  - `runs/official-eqr-sudoku-repro-20260623/score.json`
  - `runs/official-eqr-maze-repro-20260623/score.json`
- official EqR mixer/FutureSeed replacement boundary records:
  - `runs/official-eqr-mixer-replacement-triton-e1024-matched-eval-20260624T1015Z-ce2e76b/score.json`
  - `runs/rwkv_native_fs_sudoku_gate_20260624T121426Z_cb1b5a1-e64-d16b1/score.json`
- standalone GDN/FutureSeed official Sudoku records:
  - D128/L6 no-FS vs FS 600
  - D192/L10 no-FS vs FS 600
  - D192/L10 FS 1500 and 3000
  - D256 capacity, hard curriculum, exact-margin, loop8, expand_v2, feedback-attractor, latent-noise0.01

The script parses `futureseed_loop_seed*.json`, `score.json`, and `logs/run.log`.

## 4. Environment

- local Python: system `python3`
- no CUDA execution
- no AIStation access required
- no CPU smoke of model code

## 5. Commands

```bash
python3 scripts/build_loss_alignment_report.py \
  --repo . \
  --out-prefix research/reports/loss_alignment_20260630

python3 -m py_compile scripts/build_loss_alignment_report.py
```

## 6. Artifacts

- script: `scripts/build_loss_alignment_report.py`
- markdown report: `research/reports/loss_alignment_20260630.md`
- json report: `research/reports/loss_alignment_20260630.json`
- html report: `research/reports/loss_alignment_20260630.html`

## 7. Results

Official EqR reproduction facts:

| target | reproduced result |
|---|---:|
| Sudoku quick top1 exact | `0.9917` |
| Sudoku quick top4 exact | `0.9868` |
| Sudoku quick majority exact | `0.9873` |
| Maze D16/B1 exact | `0.8270` |
| Maze D64/B1 exact | `0.8920` |
| Maze D64/B128 convergence exact | `0.9444` |

GDN/FutureSeed alignment:

| comparison | CE delta | exact delta | blank delta | interpretation |
|---|---:|---:|---:|---|
| D128/L6 no-FS -> FS 600 | `-0.5688` | `+0.0107` | `+0.2239` | FutureSeed opens GDN |
| D192/L10 no-FS -> FS 600 | `-0.5890` | `+0.0176` | `+0.2347` | stronger opening evidence |
| D192/L10 FS 1500 -> 3000 | `-0.0242` | `+0.0005` | `+0.0081` | soft learning continues, exact flat |
| D192/L10 FS 1500 -> latent-noise0.01 1500 | `-0.0014` | `0.0000` | `+0.0014` | EqR-scale noise is healthy but not a plateau breaker |

Loop shape:

- D192/L10 FS 1500: loop1 exact `0.0054`, loop2 `0.0293`, loop5 `0.0293`.
- D192/L10 FS 3000: loop1 exact `0.0044`, loop2 `0.0283`, loop5 `0.0298`.
- D192/L10 loop8 1500: loop3 through loop8 exact all `0.0293`.
- D192/L10 latent-noise0.01: loop2 `0.0288`, loop3-loop5 `0.0293`.

Official EqR loss audit:

- EqR uses token LM loss plus halt BCE: `lm_loss + 0.5 * q_halt_loss`.
- EqR default training noise is `noise_scale: 0.01`.
- Sudoku quick eval uses `noise_scale=0.5`; Maze released checkpoint needs `noise_scale=0.01`.
- Maze `noise_scale=0.5` gives misleading token accuracy but bad exact.

## 8. Conclusions

Decision: mark this as a completed diagnostic and use it to narrow the next experiment.

Solid conclusions:

- FutureSeed is not dead. It is the clean opening mechanism for standalone GDN on official Sudoku.
- EqR is not winning because of a complex human-prior loss. Its training signal is simple, but its architecture/state dynamics and recurrent noisy attractor are stronger.
- Same-config longer training, latent noise, exact margin, feedback corruption, loop8, and small state tweaks all fail in the same way: blank accuracy and CE improve, full-board exact barely moves.
- The bottleneck is the conversion from per-cell confidence to board-level consistency after early loops. Later loops mostly copy the same operating point.

Next decision:

Do not run more noise-scale, margin, feedback, gate, seed, or same-config long-training tables. The next high-ROI experiment should either:

1. scale a genuinely different state-capacity axis, not just token width; or
2. change the basic recurrent state formulation so later loops preserve uncertainty and keep revising instead of freezing after loop3.

Both options need eval CE by loop / state-delta diagnostics added before training, so we do not keep judging only final exact.

## 9. Submission Record

No model submission, no checkpoint tag, no leaderboard quality update. This is a diagnostic/report artifact only.
