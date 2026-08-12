# P-LOOP-001 Provenance

- R1: `p-loop-001-gradient-conflict-20260812T135105Z-8d933dc`
  - source: clean detached pushed SHA `8d933dc547f5bfff79b1e6ca028f91b18cc55aef`
  - result: non-science engineering abort before gradient measurement
  - cause: autograd provenance walker retained object IDs instead of node proxies
- R2: `p-loop-001-gradient-conflict-r2-20260812T140000Z-f99b4a7`
  - source: clean detached pushed SHA `f99b4a7b007092a2d6289b4e619d594319232547`
  - GPU: CUDA index 0, A100-SXM4-80GB, UUID `GPU-0da20a4f-5e67-e47d-7aab-8c6efa2864ad`
  - parent checkpoint SHA256: `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
  - pinned FLA source SHA: `9c8e42e762fce087c27b673af4922795d9edb85e`
  - status: 0, no abort, no optimizer step, no checkpoint
  - result JSON SHA256: `528adc5a77740d73b59d89780651449c65a50bbb2eaaef680a601455b73aef19`
  - runtime: 340.05 seconds; peak allocation 3,502.73 MiB

The R2 JSON is the canonical result. `r2-remote-hashes.sha256` preserves the
absolute remote paths from the formal run; `manifest.sha256` hashes the copied
repository evidence.
