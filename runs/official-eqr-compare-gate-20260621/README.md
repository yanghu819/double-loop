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

## Current Blocker

GPU1 has CUDA/PyTorch and FlashAttention, but official EqR training requires
`adam_atan2_backend`. On the current AIStation image:

```text
GPU: NVIDIA A100-SXM4-80GB, GPU1 only
torch 2.7.0+cu126
flash_attn OK
hydra OK after isolated venv install
adam_atan2 OK after isolated venv install
adam_atan2_backend MISSING
```

The official README explicitly says a Python-only `adam-atan2` install is not
sufficient for training. Reinstall attempts failed because the published
`adam-atan2` sdists report inconsistent metadata versions, and the backend did
not get installed.

Official data download is also blocked in this environment:

```text
huggingface_hub snapshot_download(locuslab/EqR-data)
-> Connection to huggingface.co timed out
```

So the official comparison has been prepared, but not trained.

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
- `logs/prepare_official_compare.log`
- `logs/prep.log`
- `logs/adam_backend.log`
- `logs/adam_backend_pypi.log`
- `logs/adam_002.log`
- `logs/download_data.log`

## Decision

Do not claim official EqR has been reproduced yet.

Do not run a nominal "official" baseline by silently replacing `AdamATan2` with
AdamW; that would make the baseline less faithful. The next valid step is to
obtain or build a working `adam_atan2_backend` wheel for this CUDA/PyTorch
environment, and make `locuslab/EqR-data` reachable from the GPU environment or
pre-stage the official data under the project directory.

Once the optimizer gate passes, run one matched pair:

```bash
CUDA_VISIBLE_DEVICES=0 MAX_STEPS=500 GLOBAL_BATCH_SIZE=128 \
  scripts/official_eqr_compare/run_official_eqr_compare.sh train-base

CUDA_VISIBLE_DEVICES=0 MAX_STEPS=500 GLOBAL_BATCH_SIZE=128 \
  FUTURE_SEED_SCALE=1 \
  scripts/official_eqr_compare/run_official_eqr_compare.sh train-futureseed
```

If the 500-step sanity pair is healthy, extend to the official budget axis
rather than adding local maze-specific tricks.
