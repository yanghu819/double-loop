# P-FS3-003: Address-Local Producer Update Routing

## 1. Metainfo

- Status: approved; CUDA contract and one-step full-stack probe pending
- Date: 2026-08-07
- Branch: `codex/fs3-address-local-update-20260807`
- Benchmark: official/full-diversity hard 9x9 Sudoku
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent mechanism: D256/L12 position-QK GDN3 plus native FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Frozen matched control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

The formal source SHA is the pushed commit used to build the clean detached
AIStation worktree. It must be recorded in the run-local config and may not be
substituted after the CUDA contract.

## 2. Hypothesis

P-FS3-002 learned a nonzero residual from the real producer update, but its
single-payload bottleneck pooled all 32 K address rows with nearly uniform
attention. Exact and mixed closure did not improve, all hard-range blank
accuracies fell, and cost exceeded the registered limits. This leaves a
specific structural question: is FutureSeed losing the address-conditioned
organization of newly written evidence?

P-FS3-003 predicts that the producer update should remain in its native KxV
layout. A receiver should decide how much of each address row to emphasize or
suppress without pooling rows, remapping the basis, or decoding new payloads.
If address preservation is the missing interface, dynamic row-local routing
should improve late-loop hard closure while remaining much cheaper than the
cell-wise codec.

## 3. Configuration

For receiver layers 2 and later, let `T` be the producer terminal state, `I`
the exact initial state consumed by that producer, and `D=T-I`. The shared
router computes five generic statistics independently for each K row:

1. signed terminal-row mean after global terminal-RMS normalization;
2. log terminal-row RMS;
3. signed update-row mean after the same normalization;
4. log update-row RMS;
5. terminal/update row cosine.

A feature-sized `5 -> 5 -> 1` MLP produces `g[row]`. The candidate seed is
`T + tanh(g[row]) * D[row,:]`. The final projection is zero initialized, so
the complete candidate is bit-exact to terminal FutureSeed before training.
The mechanism is shared across all layers and heads, adds exactly 35
parameters, is equivariant to K-row and V-column permutations, and bounds every
residual element by the corresponding producer-update element.

All other variables remain frozen: D256/L12/H8/K32/V32, 12 pinned official-FLA
GDN2/Triton layers, position-QK addressing, native FutureSeed, BF16, loop5
all-loop CE, effective batch128, random traversal, optimizer state, RNG, data
order, curriculum, LR, loss, and seed52. The candidate alone resumes the same
step3000 parent to step3100. The P-FS3-001 terminal step3100 arm is the frozen
control and is not rerun.

## 4. Environment

- SSH alias: `aistation-task-gpu1`
- Persistent root: `/huyang2/double-loop`
- Python: `/opt/conda/bin/python`
- CUDA visibility: exactly index0 and the registered UUID
- FLA source SHA: `9c8e42e762fce087c27b673af4922795d9edb85e`
- FLA backend dispatch disabled; Triton convolution required
- Source: pushed SHA in a clean detached worktree
- CPU model smoke and all GPU overlap are forbidden

## 5. Commands And Preflight

The strict contract is:

```bash
CUDA_VISIBLE_DEVICES=0 \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
TRITON_CACHE_DIR=/huyang2/double-loop/.cache/triton \
TORCHINDUCTOR_CACHE_DIR=/huyang2/double-loop/.cache/torchinductor \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
TMPDIR=/huyang2/double-loop/.cache/tmp \
FLA_DISABLE_BACKEND_DISPATCH=1 FLA_CONV_BACKEND=triton \
/opt/conda/bin/python \
experiments/rwkv_fs_sudoku/check_futureseed_address_local_update_cuda.py
```

It must prove the exact GPU UUID and one visible device, pinned FLA SHA,
12 official `GatedDeltaNet2` layers, `ChunkGDN2FunctionBackward`, position-QK,
bit-exact output and every terminal state at zero initialization, exact
three-tensor checkpoint migration, 35-parameter delta, nonzero gradients for
all router tensors after opening the output, independent K/V permutation
equivariance, and the elementwise update bound.

Only after that contract passes, run one exact full-stack step3000-to3001 probe:

```bash
CUDA_VISIBLE_DEVICES=0 FS3_FULL_STACK_PROBE=1 \
scripts/run_futureseed_address_local_update_arm.sh
```

The probe must accept exact resume, stop exactly at step3001, save a complete
checkpoint and metrics JSON, exercise the official CUDA path, and show finite
nonzero router/output gradients. The formal command is the same launcher
without `FS3_FULL_STACK_PROBE=1`.

## 6. Artifacts

Pending contract/probe/formal execution. Every log, status, resolved config,
source snapshot, checkpoint, metrics JSON, same-board export, and SHA256 must
remain below `/huyang2/double-loop` and be recorded here after completion.

## 7. Registered Decision Gates

Activation at step3100 requires all of:

- mean absolute row gain at least `1e-4`;
- finite nonzero within-state K-row gain standard deviation;
- finite nonzero between-board gain standard deviation;
- finite nonzero residual relative RMS and between-board residual variation.

Primary quality passes only if hard official51-64 macro loop5 exact improves
by at least `+0.02` over the frozen control and no official 51-55, 56-60, or
61-64 loop5 blank accuracy regresses by more than `0.01`.

The alternate route requires mixed loop5 exact at least `+0.03`, non-regressive
official61-64 exact, and stronger same-board loop3-to5 wrong-cell correction.
Fresh-process elapsed overhead and peak allocated-memory overhead must both be
below 10 percent. Integrity, activation, quality, and cost are conjunctive.

## 8. Decision Boundary

Any contract, integrity, activation, quality, or cost miss discards this exact
address-local router. Do not rescue it with row features, hidden width, bias,
gain scale, normalization, layer/head-specific parameters, extra steps, seed,
LR, loss, batch, model width/depth, payload count, or a nearby address variant.
The next decision would instead move to a stronger generic recurrent
memory/address/state update mechanism.

## 9. Submission Record

Not applicable. This is a mechanism experiment, not a platform submission.
