# official-eqr-futureseed-mixer-replacement-sudoku-20260624

## 1. Metainfo

- Plan ID: `P-EQR-013`
- Status: implementation ready, pending GPU1 launch
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

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not applicable.
