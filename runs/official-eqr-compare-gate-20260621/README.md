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

## Current Blocker

Official EqR still has not produced a completed train/eval result.

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

After those fixes, the base run reaches training step 0:

```text
run_name=official-eqr-base-direct-e64-20260621T052644Z-aba94e9
epochs: 64
Estimated total training steps: 448
Using optimizer: adam_atan2
Training [l94]:   0%|          | 0/448 [00:00<?, ?it/s, loss=0]
```

It then stalls in unkillable `D` state:

```text
PID 7302 wchan: cxiWaitEventWait
GPU: 503 MiB / 81920 MiB, 0% util
SIGKILL did not remove the process
```

This is now an AIStation/container runtime blocker, not a missing Python
package or missing Hugging Face data issue. GPU1 should be restarted before the
next official compare attempt.

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

## Decision

Do not claim official EqR has been reproduced yet.

Do not run a nominal "official" baseline by silently replacing `AdamATan2` with
AdamW; that would make the baseline less faithful.

The next valid step is to restart GPU1, verify the uploaded offline packages and
data are still under `/huyang2/double-loop/official_eqr_compare`, then rerun the
direct-Python official base sanity. If it passes, run the matched FutureSeed
sanity under the same launcher and budget.

Current short-run command shape:

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=64 TRAIN_EPOCHS_PER_ITER=64 GLOBAL_BATCH_SIZE=128 \
  scripts/official_eqr_compare/run_official_eqr_compare.sh train-base

CUDA_VISIBLE_DEVICES=0 EPOCHS=64 TRAIN_EPOCHS_PER_ITER=64 GLOBAL_BATCH_SIZE=128 \
  FUTURE_SEED_SCALE=1 \
  scripts/official_eqr_compare/run_official_eqr_compare.sh train-futureseed
```

If the 500-step sanity pair is healthy, extend to the official budget axis
rather than adding local maze-specific tricks.
