# Official EqR Fair-Compare Gate

Created: `2026-06-21`

## Question

Can we compare FutureSeed+loop against EqR on the official EqR codebase, instead
of using the local no-Hydra online maze probe?

## Correction To Earlier Evidence

Earlier Maze31 results used `scripts/eqr_maze_probe.py`, which imports
`EqRModel` from a patched EqR clone but uses a local online perfect-maze sampler,
local loss/metrics, and local launch path. That is useful for mechanism probing,
but it is not a faithful reproduction of official EqR experiments.

The fair comparison must use the official EqR pipeline:

- official repo: `https://github.com/locuslab/eqr`
- official SHA: `aba94e9cde0f273ce644db5261cd6915ba6561f0`
- official task/config: `eqr_maze_unique`
- official dataset path: `data/maze-30x30-unique-1k`
- official training entrypoint: `scripts/train.sh eqr_maze_unique`

## Implemented Gate

Added reusable scripts:

- `scripts/official_eqr_compare/run_official_eqr_compare.sh`
- `scripts/official_eqr_compare/apply_futureseed_patch.py`

The launcher creates two isolated clones under:

```text
/huyang2/double-loop/official_eqr_compare/eqr-clean
/huyang2/double-loop/official_eqr_compare/eqr-futureseed
```

`eqr-clean` is pristine official EqR. `eqr-futureseed` is the same official SHA
plus the smallest FutureSeed change: a learned gated projection from `z_H` to
`z_L` before L recursion and from `z_L` to `z_H` before H update. It does not add
selector, repair, search, task-specific rules, online maze generation, or custom
losses.

## Offline Staging Update

Update time: `2026-06-21T05:24:43Z`.

The external-data blocker was removed by downloading locally and uploading to
GPU1:

- local Hugging Face snapshot: `locuslab/EqR-data`
- uploaded archive: `/huyang2/double-loop/official_eqr_compare/artifacts/eqr-data-full.tgz`
- extracted data: `/huyang2/double-loop/official_eqr_compare/eqr-clean/data`
- extracted size: `1.3G`
- archive SHA256: `1875b5e14fd6ca8b5bacbe2c0fd09efc661faeb7a1d44cff3d530b09525bb293`

The Python dependency wheelhouse was also downloaded locally and uploaded:

```text
remote archive: /huyang2/double-loop/official_eqr_compare/artifacts/wheelhouse.tgz
remote extracted wheelhouse size: 23M
archive SHA256: 4c4c7c3cbfb55e42aa74f94583b0b6dcc430fee9efab3943992cda22d7551423
```

Notable wheels staged locally before upload:

- `hydra-core==1.3.3`
- `omegaconf==2.3.1`
- `coolname==5.0.0`
- `argdantic==1.3.3`
- `antlr4-python3-runtime==4.9.3`
- `adam-atan2==0.0.3`
- `setuptools_scm==9.2.2`
- `nvidia-cuda-nvcc-cu12==12.6.85`
- `exceptiongroup==1.3.0`

`nvidia-cuda-nvcc-cu12` did not provide `bin/nvcc`; both `12.6.85` and a local
probe of `12.9.86` only exposed `ptxas`/headers. Since the AIStation image only
has `/usr/local/cuda-11.7/bin/nvcc` while PyTorch is `2.7.0+cu126`, an exact
CUDA 12.6 extension build was not possible from wheels alone.

To keep the official optimizer rather than silently swapping to AdamW, a
dependency-only fallback build was used:

- source downloaded locally: `https://github.com/imoneoi/adam-atan2`
- source SHA: `42a1aeb38cd0838d42631b18369505e86e5bd3fe`
- uploaded archive: `/huyang2/double-loop/official_eqr_compare/artifacts/adam-atan2-src.tgz`
- archive SHA256: `fe915f70dd8350d1f4c17157294b0c0e591c05df988d5734701e4ac41c7a97d6`
- compile fallback: A100-only `sm80`, skip PyTorch CUDA-version guard, use system
  CUDA 11.7 nvcc

This changes only the dependency build path and target architecture; it does
not alter EqR, FutureSeed, dataset, optimizer algorithm, selector, repair,
search, or task rules.

The backend gate now passes:

```text
AdamATan2 <class 'adam_atan2.adam_atan2.AdamATan2'>
adam_atan2_backend /huyang2/double-loop/official_eqr_compare/.venv/lib/python3.10/site-packages/adam_atan2_backend.cpython-310-x86_64-linux-gnu.so
```

## Official Step448 Pair

The official launcher needed wrapper fixes:

- `torchrun` used system Python, so venv site-packages had to be exposed through
  `PYTHONPATH`.
- short runs must override official fields as `epochs` and
  `train_epochs_per_iter`; `max_steps` is not present in the YAML and must not
  be used as a normal override.
- absent optional fields need Hydra `+` syntax, e.g. `+wandb_mode=disabled`.
- W&B offline/core stalled in this container, so W&B is disabled for the
  sanity run.
- single-GPU direct Python launching avoids the local `torchrun`/distributed
  wait path and is appropriate because official `pretrain.py` only initializes
  distributed when `LOCAL_RANK` is present.
- the installed `flash-attn` package is ABI-incompatible with the container's
  PyTorch, so the launcher applies a runtime-only attention fallback to PyTorch
  SDPA when FlashAttention import fails. This fallback is applied equally to
  `eqr-clean` and `eqr-futureseed`; it changes the attention kernel path, not
  the task, loss, optimizer, dataset, selector, repair, search, or model claim.

After these fixes, the official short-budget pair completed on GPU1:

```text
base run:
  official-eqr-base-sdpa-e64-20260621T053554Z-aba94e9
  checkpoint: outputs/base/outputs/fk7ginoq/2026-6-21/5-31-59/checkpoints/step_448_fk7ginoq.pth
  final train loss: 0.466513

futureseed run:
  official-eqr-futureseed-sdpa-e64-20260621T053908Z-aba94e9
  checkpoint: outputs/futureseed/outputs/5vl2vfof/2026-6-21/5-35-15/checkpoints/step_448_5vl2vfof.pth
  final train loss: 0.504108
```

Both were evaluated with the official `evaluate.py` entrypoint and the official
`config/eval/depth_breadth.yaml` settings:

```text
halt_max_steps: 16
noise_scale: 0.5
init_std: 1.0
global_batch_size: 128
eval batches: 8
```

Final checkpoint metrics:

| metric | clean EqR | FutureSeed | delta |
| --- | ---: | ---: | ---: |
| `all/accuracy` | `0.5158277588` | `0.5576755981` | `+0.0418478394` |
| `all/exact_accuracy` | `0.0` | `0.0` | `+0.0` |
| `all/lm_loss` | `1.4151149902` | `1.3903227539` | `-0.0247922363` |
| `all/total_loss` | `1.4784158936` | `1.4536179199` | `-0.0247979736` |
| `all/residual_of_1_steps` | `593.314` | `587.955` | `-5.359` |
| `all/residual_of_8_steps` | `471.817` | `482.472` | `+10.655` |
| `all/residual_of_16_steps` | `436.811` | `437.060` | `+0.249` |

Interpretation:

- This is the first completed official-code, official-data EqR vs FutureSeed
  gate.
- FutureSeed has a real positive short-budget signal at token/loss level:
  `+4.18` accuracy points and lower loss.
- It is not solved reasoning: exact accuracy is still zero for both.
- Extra loops are not yet the source of the win: FutureSeed is better at loop1
  residual but not at loop8 or loop16 residual.

## Remote State

Prepared on GPU1 under:

```text
/huyang2/double-loop/official_eqr_compare
```

Verified clone state:

```text
eqr-clean HEAD        aba94e9cde0f273ce644db5261cd6915ba6561f0
eqr-futureseed HEAD   aba94e9cde0f273ce644db5261cd6915ba6561f0
eqr-futureseed diff   2 files changed, 45 insertions
```

Pulled artifacts:

- `artifacts/futureseed.patch`
- `artifacts_after_offline/offline_transfer_manifest.txt`
- `logs/prepare_official_compare.log`
- `logs/prep.log`
- `logs/adam_backend.log`
- `logs/adam_backend_pypi.log`
- `logs/adam_002.log`
- `logs/download_data.log`
- `logs_after_offline/logs/adam_backend_source_build.log`
- `logs_after_offline/logs/adam_backend_source_build_nvcc12.log`
- `logs_after_offline/logs/adam_backend_cu117_fallback_build.log`
- `logs_after_offline/logs/official-eqr-base-direct-e64-20260621T052644Z-aba94e9.log`
- `artifacts_step448/official-eqr-pair-step448-20260621T0541Z.tgz`
- `artifacts_step448/logs/official-eqr-base-sdpa-e64-20260621T053554Z-aba94e9.log`
- `artifacts_step448/logs/official-eqr-futureseed-sdpa-e64-20260621T053908Z-aba94e9.log`
- `artifacts_step448/logs/eval-official-base-final-step448-20260621T0537Z.log`
- `artifacts_step448/logs/eval-official-futureseed-final-step448-20260621T0538Z.log`
- `artifacts_step448/outputs/**/eval_metrics_step_448.json`
- `score_step448.json`
- `leaderboard_step448.csv`
- `official_eqr_pair_step448.png`
- `index.html`

## Decision

We can now claim a completed official EqR sanity reproduction under a short
budget, with one explicit runtime kernel fallback. We should not claim that
FutureSeed+loop beats EqR as a reasoning method yet.

The result supports a narrower and useful claim:

```text
FutureSeed improves official EqR optimization under the same short budget, but
the current FutureSeed insertion has not made later loops perform correction.
```

The next valid experiment is not another local maze-probe table. It should use
the official EqR code path and scale the budget or loop-specific readout:

- longer official-budget run to see whether the token/loss advantage becomes
  exact accuracy;
- or a targeted official eval that reports whether increasing halt steps
  reduces residual/exact failures rather than merely changing the operating
  point.

Current short-run command shape:

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=64 TRAIN_EPOCHS_PER_ITER=64 GLOBAL_BATCH_SIZE=128 \
  scripts/official_eqr_compare/run_official_eqr_compare.sh train-base

CUDA_VISIBLE_DEVICES=0 EPOCHS=64 TRAIN_EPOCHS_PER_ITER=64 GLOBAL_BATCH_SIZE=128 \
  FUTURE_SEED_SCALE=1 \
  scripts/official_eqr_compare/run_official_eqr_compare.sh train-futureseed
```

Official final eval command shape:

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_GLOBAL_BATCH_SIZE=128 \
  scripts/official_eqr_compare/run_official_eqr_compare.sh eval-base <checkpoint.pth>

CUDA_VISIBLE_DEVICES=0 EVAL_GLOBAL_BATCH_SIZE=128 \
  scripts/official_eqr_compare/run_official_eqr_compare.sh eval-futureseed <checkpoint.pth>
```
