# official-eqr-futureseed-mixer-replacement-sudoku-20260624

## 1. Metainfo

- Plan ID: `P-EQR-013`
- Status: done
- Machine: AIStation `GPU1` only
- Local branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

The previous cheap-bidirectional probe was not the clean paper experiment: it
added a future source to a causalized EqR branch. The clean experiment is to
keep the official EqR codebase and replace the ReasoningBlock token mixer with
a generic FutureSeed recurrent mixer.

If FutureSeed is a cheap bidirectional information mechanism, a
FutureSeed-scan mixer should approach the official EqR mixer baseline under a
matched small Sudoku budget, or show a clear early optimization advantage. If it
cannot compete even on this quick gate, the current FutureSeed replacement is
not ready as the main paper method.

## 3. Configuration

Compare exactly two arms:

- `mixer-base`: official EqR Sudoku mixer baseline, using upstream
  `train/eqr_sudoku` semantics. This means `arch.mlp_t=true` unless explicitly
  overridden by the official config.
- `futureseed-mixer`: same official EqR codebase and config, but
  `ReasoningBlock` token mixing is replaced by `FutureSeedScanMixer`, selected
  with `arch.mixer_replacement_mode=future_seed_scan`.

Shared quick gate:

- Official dataset: `sudoku-extreme-1k-aug-1000`
- `arch.hidden_size=192`
- `arch.num_heads=6`
- `arch.halt_max_steps=16`
- `arch.noise_scale=0.01`
- `global_batch_size=128`
- `epochs=256`
- `eval_interval_steps=250`
- `checkpoint_interval_steps=250`

No Sudoku solver, no repair, no selector, no oracle rollout, no task-specific
rule. No seed sweep.

## 4. Environment

Remote project root: `/huyang2/double-loop`

Official EqR compare root:
`/huyang2/double-loop/official_eqr_compare`

The implementation adds:

- `scripts/official_eqr_compare/apply_futureseed_mixer_replacement_patch.py`
- `prepare-mixer-replacement`
- `train-mixer-base`
- `train-futureseed-mixer`

## 5. Commands

The launch script will run detached from a GitHub-pushed SHA:

```bash
RUN_STAMP=<timestamp>-mixerreplace-<sha> \
EQR_TRAIN_CONFIG=train/eqr_sudoku \
EQR_DATA_LINKS=sudoku-extreme-1k-aug-1000=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/eqr-official/data/sudoku-extreme-1k-aug-1000 \
GLOBAL_BATCH_SIZE=128 EPOCHS=256 TRAIN_EPOCHS_PER_ITER=256 \
EVAL_INTERVAL_STEPS=250 CHECKPOINT_INTERVAL_STEPS=250 \
EQR_EXTRA_OVERRIDES='arch.hidden_size=192 arch.num_heads=6 arch.halt_max_steps=16 arch.noise_scale=0.01' \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/official_eqr_compare/run_official_eqr_compare.sh prepare-mixer-replacement
```

Then run `train-mixer-base` and `train-futureseed-mixer` sequentially, parse
paired step250/500 metrics, and truncate if the replacement is clearly losing.

## 6. Artifacts

- Source SHA for vectorized replacement: `735fa2b4df06e42039f28dafbf172295644bb14c`
- Source SHA for Triton CUDA scan backend:
  `d9c2600e47d0b5e0950bf884ba8a155de3e6e198`
- Baseline SHA for official mixer-base arm: `c9b5bd76a227521489c56ea9c4a7c2847740bb56`
- Remote comparison JSON:
  `/huyang2/double-loop/official_eqr_compare/artifacts/mixer_replacement_sudoku_20260624T0610Z-mixerreplace-735fa2b-vectorized/comparison_step250_500.json`
- Local comparison JSON:
  `runs/official-eqr-futureseed-mixer-replacement-sudoku-20260624/comparison_step250_500.json`
- Triton CUDA backend microcheck:
  `runs/fs-triton-backward-check-20260624-d9c2600/result.json`
- Triton train-only backend viability run:
  `runs/official-eqr-fs-mixer-triton-sudoku-trainonly-20260624T0930Z-d9c2600/`
- Matched CUDA/Triton tiny eval gates:
  `runs/official-eqr-mixer-replacement-triton-matched-eval-20260624T0940Z-ce2e76b/`,
  `runs/official-eqr-mixer-replacement-triton-e256-matched-eval-20260624T0955Z-ce2e76b/`,
  `runs/official-eqr-mixer-replacement-triton-e1024-matched-eval-20260624T1015Z-ce2e76b/`
- Matched e1024 train-only logs:
  `runs/official-eqr-mixer-base-trainonly-e1024-20260624T1000Z-ce2e76b/`,
  `runs/official-eqr-futureseed-mixer-triton-trainonly-e1024-20260624T1010Z-ce2e76b/`
- Baseline train log:
  `/huyang2/double-loop/official_eqr_compare/logs/official-eqr-mixer-base-sudoku-20260624T0520Z-mixerreplace-c9b5bd7.log`
- FutureSeed train log:
  `/huyang2/double-loop/official_eqr_compare/logs/official-eqr-futureseed-mixer-sudoku-20260624T0610Z-mixerreplace-735fa2b-vectorized.log`
- FutureSeed step500 standalone eval JSON:
  `/huyang2/double-loop/official_eqr_compare/outputs/futureseed-mixer/outputs/177ddj3j/2026-6-24/6-9-10/eval_preds/step_500_step500_vectorized_arch_halt_max_steps-16_arch_noise_scale-0.5_arch_H_init_std-1.0_arch_L_init_std-1.0/eval_metrics_step_500.json`
- Abort records:
  `runs/official-eqr-futureseed-mixer-replacement-sudoku-20260624/abort_futureseed_compile.json`,
  `runs/official-eqr-futureseed-mixer-replacement-sudoku-20260624/abort_eval_too_slow.json`,
  `runs/official-eqr-futureseed-mixer-replacement-sudoku-20260624/abort_step500_train_eval_interrupted.json`,
  `runs/official-eqr-fs-mixer-triton-sudoku-abort-20260624T0926Z-d9c2600/abort.json`

## 7. Results

| step | arm | accuracy | exact | lm_loss | total_loss | residual1 | residual16 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 250 | official mixer-base | 0.091652 | 0.000000 | 2.554581 | 2.558680 | 243.356 | 229.549 |
| 250 | FutureSeed mixer | 0.105617 | 0.000000 | 2.483983 | 2.488077 | 233.214 | 162.895 |
| 250 | FS - base | +0.013965 | 0.000000 | -0.070598 | -0.070603 | -10.142 | -66.655 |
| 500 | official mixer-base | 0.098043 | 0.000000 | 2.522132 | 2.526218 | 242.984 | 228.938 |
| 500 | FutureSeed mixer | 0.178486 | 0.000000 | 2.339771 | 2.343806 | 262.142 | 214.241 |
| 500 | FS - base | +0.080443 | 0.000000 | -0.182361 | -0.182412 | +19.158 | -14.698 |

Important caveat: step250 is train-time eval under the training config
(`arch.noise_scale=0.01`). The original step500 train-time eval was interrupted
when the AIStation lease halted, so step500 for the FutureSeed arm comes from a
standalone official `evaluate.py` run, whose default overrides include
`arch.noise_scale=0.5`. This still shows the vectorized replacement checkpoint
is much better than baseline on token accuracy/loss, but step500 is not a
perfectly matched train-time-eval point.

Execution failures were informative:

- The first FutureSeed mixer replacement used a Python loop over sequence
  positions. `torch.compile` stalled at step0.
- Running that same Python-loop scan with `DISABLE_COMPILE=1` unblocked
  training but made official eval infeasible, around seconds per eval batch.
- Commit `735fa2b` replaced the loop with a generic parallel-prefix recurrent
  scan. This made the arm compile and run at about `3.6` official eval batches/s
  on GPU1.
- Commit `d9c2600` added a Triton CUDA custom-autograd scan backend. On GPU1
  A800, it matched the prefix fallback to numerical tolerance and ran the scan
  microcheck in `0.00527s` versus `0.06410s` for the prefix fallback after
  warmup, about `12.2x` faster.
- The Triton backend also trained through the official EqR FutureSeed mixer
  path for `448/448` Sudoku steps with final train loss `1.626443`. This was a
  backend viability run only: train-time eval was disabled after the first
  full-eval launch expanded to `3304` eval batches and was killed as low ROI.

### CUDA/Triton matched scale gate

After the CUDA scan backend was validated, I ran bounded official eval for three
matched budgets. All rows use the same tiny official eval gate:
`max_eval_steps=16`, batch `128`, `halt_max_steps=16`, `noise_scale=0.01`,
`different_init=1`.

| budget | arm | train loss | accuracy | exact | lm_loss | residual1 | residual4 | residual8 | residual16 | eval sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| step448 | official mixer-base | 2.359425 | 0.095498 | 0.000000 | 2.534932 | 238.800 | 229.108 | 229.149 | 229.125 | 8.558 |
| step448 | FutureSeed mixer | 1.626443 | 0.188079 | 0.000000 | 2.313245 | 226.281 | 154.876 | 150.535 | 150.869 | 13.012 |
| step448 | FS - base | -0.732982 | +0.092581 | 0.000000 | -0.221687 | -12.519 | -74.232 | -78.614 | -78.256 | +4.454 |
| e256/step1792 | official mixer-base | 1.120432 | 0.366766 | 0.000000 | 1.803480 | 165.761 | 63.063 | 60.984 | 61.583 | 7.541 |
| e256/step1792 | FutureSeed mixer | 1.448282 | 0.420693 | 0.000000 | 1.523322 | 188.403 | 35.650 | 13.391 | 7.973 | 15.192 |
| e256/step1792 | FS - base | +0.327850 | +0.053928 | 0.000000 | -0.280157 | +22.642 | -27.413 | -47.593 | -53.610 | +7.651 |
| e1024/step7168 | official mixer-base | 0.767679 | 0.664400 | 0.024902 | 0.766357 | 195.923 | 4.833 | 4.849 | 4.979 | 7.957 |
| e1024/step7168 | FutureSeed mixer | 1.408197 | 0.423050 | 0.000000 | 1.402906 | 198.346 | 6.381 | 3.238 | 3.065 | 14.814 |
| e1024/step7168 | FS - base | +0.640518 | -0.241350 | -0.024902 | +0.636548 | +2.422 | +1.548 | -1.611 | -1.914 | +6.857 |

This is the key decision point. The CUDA FutureSeed mixer wins clearly at
step448 and still wins token/loss at e256. By e1024, the official mixer-base
opens exact accuracy and dominates accuracy/loss. FutureSeed still has lower
late-loop residual at e1024, but that is no longer a positive sign: it means the
recurrent state has converged to a stable wrong attractor.

## 8. Conclusions

Mixed, and now much more precise.

Positive part: FutureSeed-as-mixer-replacement is real enough to test. The
Python loop was invalid for the claim, but the Triton CUDA scan is numerically
checked and fast enough. Under bounded official eval, FutureSeed gives a strong
early sample-efficiency and state-contraction signal at step448 and e256.

Negative part: this exact replacement is not a long-budget win. At e1024 the
official mixer-base reaches nonzero exact (`0.024902`) and much better token
accuracy (`0.664400` vs `0.423050`). The FutureSeed arm's lower residual8/16 is
a warning, not a success: it stabilizes faster, but around the wrong answer.

Decision: stop simply extending this replacement. The next high-ROI work is a
modeling change that keeps the FutureSeed idea but prevents the stable-wrong
attractor. Two clean options remain within the bitter-lesson boundary:

- use FutureSeed as a cheap future-context source or initialization while
  retaining enough learned noncausal mixing/normalization capacity for final
  decisions;
- add a simple generic gated/normalized FutureSeed state update so loop can
  keep moving decision boundaries instead of collapsing into a fixed wrong
  state.

Do not respond by running seeds, hidden-size tables, gate-bias tables, or longer
training of the same replacement.

## 9. Submission Record

Not applicable.
