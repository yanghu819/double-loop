# official-eqr-rwkv-native-futureseed-mixer-sudoku-20260624

## 1. Metainfo

- Plan ID: `P-EQR-015`
- Status: discarded as positive evidence
- Machine: AIStation `GPU1` only
- Local branch: `codex/gpu1-experiment-tracking`
- Remote work dir: `/huyang2/double-loop/official_eqr_compare_shared`
- Source SHA: `cb1b5a14292cb97e88bbc0f3d5f206a180725f79`
- Official EqR SHA: `aba94e9cde0f273ce644db5261cd6915ba6561f0`

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

Final remote launcher shape:

```bash
CUDA_VISIBLE_DEVICES=0 \
EQR_TRAIN_CONFIG=train/eqr_sudoku \
GLOBAL_BATCH_SIZE=128 EPOCHS=64 TRAIN_EPOCHS_PER_ITER=64 \
EVAL_INTERVAL_STEPS=999999 CHECKPOINT_INTERVAL_STEPS=250 \
DISABLE_COMPILE=1 \
RWKV_MIXER_KERNEL=statepassing RWKV_MIXER_LAYERS=2 \
EQR_EXTRA_OVERRIDES='arch.hidden_size=192 arch.num_heads=6 arch.halt_max_steps=16 arch.noise_scale=0.01' \
bash scripts/official_eqr_compare/run_official_eqr_compare.sh train-rwkv-mixer
```

Then run `train-rwkv-native-fs-mixer` with the same settings. Intermediate
train-time eval was disabled because the default official full eval is too
expensive for this gate.

Evaluation used official `evaluate.py` with a bounded gate:

```yaml
halt_max_steps: 16
noise_scale: 0.5
init_std: 1.0
global_batch_size: 128
max_eval_steps: 16
different_init: 1
```

The heavier official `sudoku_lite_noise05.yaml` (`D64/B128`) was attempted for
no-FS, but was killed as low ROI after it remained on the first eval batch for
about six minutes. The bounded `D16/B1` gate was then used for both arms.

## 6. Artifacts

Remote artifact root:
`/huyang2/double-loop/official_eqr_compare_shared/artifacts/rwkv_native_fs_sudoku_gate_20260624T121426Z_cb1b5a1-e64-d16b1`

Local artifact root:
`runs/rwkv_native_fs_sudoku_gate_20260624T121426Z_cb1b5a1-e64-d16b1`

Included artifacts:

- `launch.env`
- `hypothesis.txt`
- `eval_sudoku_d16_b1_noise05.yaml`
- `rwkv-mixer.train.log`
- `rwkv-native-fs-mixer.train.log`
- `rwkv-mixer.eval.log`
- `rwkv-native-fs-mixer.eval.log`
- `rwkv-mixer.eval_metrics_step_448.json`
- `rwkv-native-fs-mixer.eval_metrics_step_448.json`
- `score.json`
- `README.md`
- `index.html`

Checkpoints remain only on the remote filesystem and are not committed.

## 7. Results

| arm | train final loss | eval acc | exact | eval lm loss | residual16 | eval time |
|---|---:|---:|---:|---:|---:|---:|
| RWKV no-FS | 1.592871 | 0.165159628 | 0.000000 | 2.370409 | 222.312012 | 74.67s |
| RWKV native-FS | 1.590624 | 0.165147573 | 0.000000 | 2.369608 | 222.399414 | 124.47s |

Delta FS minus no-FS:

- accuracy: `-0.000012055`
- exact: `+0.000000`
- lm loss: `-0.000801325`
- total loss: `-0.000808001`
- residual16: `+0.087402`
- train final loss: `-0.002247`
- eval time ratio: `1.667x`

## 8. Conclusions

Discard as positive evidence.

This result is useful because it validates the corrected implementation path:
native FutureSeed-RWKV is not the earlier right-to-left scan, it uses CUDA
statepassing, and it trains in the official EqR codebase. But it does not give
the needed mechanism win. At matched early budget it is essentially tied with
the no-FS RWKV control, has zero exact accuracy like the control, does not
improve residual16, and costs more eval time.

Do not run seed/gate/temperature sweeps for this exact mechanism. If we continue
this line, the next experiment must change generic state dynamics so the
terminal seed actually changes later recurrent computation; otherwise move back
to official EqR baselines and stronger proxy selection.

## 9. Submission Record

Not applicable.
