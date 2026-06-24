# official-eqr-rwkv-native-futureseed-mixer-sudoku-20260624

## 1. Metainfo

- Plan ID: `P-EQR-015`
- Status: in progress
- Machine: AIStation `GPU1` only
- Local branch: `codex/gpu1-experiment-tracking`
- Remote work dir: `/huyang2/double-loop/official_eqr_compare`

## 2. Hypothesis

FutureSeed should be tested in its native form, not as a right-to-left scan.
The mechanism claim is: a recurrent RWKV token mixer can use the previous
layer's terminal recurrent state as the next layer's initial state, giving a
cheap path for global information to affect later recurrent computation.

Prediction: under the same official EqR Sudoku data/eval budget, RWKV +
native FutureSeed should beat the same RWKV mixer without FutureSeed on early
loss/accuracy/residuals, and ideally close part of the gap to the official EqR
mixer baseline. If it does not beat no-FS, this mechanism is not ready to be a
paper mainline inside official EqR.

## 3. Configuration

- Official EqR upstream SHA: `aba94e9cde0f273ce644db5261cd6915ba6561f0`
- Dataset: official `sudoku-extreme-1k-aug-1000`
- Train config: `train/eqr_sudoku`
- Shared budget gate: `hidden_size=192`, `num_heads=6`, `halt_max_steps=16`,
  `noise_scale=0.01`, `global_batch_size=128`
- Arms:
  - `official-mixer-base`: official EqR Sudoku mixer
  - `rwkv-mixer`: official `ReasoningBlock` token mixer replaced by forward
    RWKV7 recurrent token mixer, no FutureSeed
  - `rwkv-native-fs-mixer`: same RWKV mixer, plus terminal recurrent state from
    RWKV layer `i` seeds RWKV layer `i+1`
- RWKV kernel target: `statepassing` CUDA path; if unavailable, stop and record
  the backend blocker rather than interpreting a slow fallback as final.

No reverse scan, no solver, no repair, no selector, no oracle rollout, no
Sudoku rule, no seed sweep.

## 4. Environment

The patch script copies the vendored RWKV7 CUDA wrapper from this repo into the
fresh official EqR clone and patches `models/eqr.py` plus `config/arch/eqr.yaml`.
All caches and compiled extensions stay under
`/huyang2/double-loop/official_eqr_compare/.cache`.

Static local checks passed before remote launch:

- `python -m py_compile scripts/official_eqr_compare/apply_rwkv_native_futureseed_mixer_patch.py`
- `bash -n scripts/official_eqr_compare/run_official_eqr_compare.sh`
- Patch applied cleanly to official upstream SHA and patched `models/eqr.py`
  passed `py_compile`

## 5. Commands

Planned remote launcher shape:

```bash
CUDA_VISIBLE_DEVICES=0 \
EQR_TRAIN_CONFIG=train/eqr_sudoku \
EQR_DATA_LINKS=sudoku-extreme-1k-aug-1000=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/eqr-official/data/sudoku-extreme-1k-aug-1000 \
GLOBAL_BATCH_SIZE=128 EPOCHS=256 TRAIN_EPOCHS_PER_ITER=256 \
EVAL_INTERVAL_STEPS=250 CHECKPOINT_INTERVAL_STEPS=250 \
RWKV_MIXER_KERNEL=statepassing RWKV_MIXER_LAYERS=2 \
EQR_EXTRA_OVERRIDES='arch.hidden_size=192 arch.num_heads=6 arch.halt_max_steps=16 arch.noise_scale=0.01' \
bash scripts/official_eqr_compare/run_official_eqr_compare.sh prepare-rwkv-native-fs
```

Then run `train-official-mixer-base`, `train-rwkv-mixer`, and
`train-rwkv-native-fs-mixer` with the same settings. The first decision gate is
paired step250/500 metrics; continue only on a clear RWKV+FS > RWKV no-FS
signal.

## 6. Artifacts

Pending remote artifact root:
`/huyang2/double-loop/official_eqr_compare/artifacts/rwkv_native_fs_sudoku_<timestamp>`.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not applicable.
