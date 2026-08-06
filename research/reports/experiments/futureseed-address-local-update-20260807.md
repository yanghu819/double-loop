# P-FS3-003: Address-Local Producer Update Routing

## 1. Metainfo

- Status: discarded; strict matched decision complete
- Date: 2026-08-07
- Branch: `codex/fs3-address-local-update-20260807`
- Formal source SHA: `8372387d2cc352f65b6b6649d963d71999e32ec3`
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
- Formal candidate:
  `p-fs3-003-address-local-s3100-20260806T231831Z-8372387`

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

## 6. Artifacts And Integrity

The pushed source was checked out as clean detached SHA `8372387` at
`/huyang2/double-loop/worktrees/p-fs3-003-8372387`. CUDA index0 exposed only
the registered UUID. The strict contract proved:

- 12/12 pinned official-FLA `GatedDeltaNet2` layers and
  `ChunkGDN2FunctionBackward` with Triton convolution;
- bit-exact output and all 12 terminal states at zero initialization;
- exactly three migrated router tensors and a 35-parameter delta;
- finite nonzero gradients for every router tensor after opening the output;
- independent K-row/V-column permutation error `2.3842e-07`;
- zero elementwise update-bound error.

The corrected step3001 full-stack probe exited0, accepted the exact parent,
saved exactly step3001, and activated the official CUDA path. Its metrics and
checkpoint SHA256 are
`9fa6f8d1c2fbad7b8b184182182faab95c00a22dd37953f54347bd22d3d4d7bb`
and `2212c6a2f5a906ea7e2de2cb7660100680aae055e4e0bebd96b10df00c26d2d6`.
The earlier contract PYTHONPATH miss and probe-wrapper quoting miss both
occurred before model construction; each has a non-science abort record and
neither contributed data.

The formal run exited0. There was no NaN, OOM, fallback, source drift, data
drift, second compute app, or GPU UUID drift. Primary candidate artifacts are:

- metrics SHA256:
  `2c6cad78427a40af7730a7f5653ccd78edd41884168c0f864d12cdb0b5fccdb7`;
- step3100 checkpoint SHA256:
  `a5fb6f355d562089758be988956f9d412d0f4f729955b7eb68f69877f3a5b281`;
- resolved config SHA256:
  `f3fad3c25345cd662073038987e5f0a6ecb222264b3e62c7ee2fcc6ac99e8cc3`;
- run log SHA256:
  `7108d0a8164765160304d179678dfe7efdadf14fda4f9e114df0b16b7af54d6c`;
- source snapshot SHA256:
  `46fdf59912b540f75b802da037e6f88f3372508500d13792a1706d52d6ecfb80`.

The machine-readable matched comparison and hardest same-board HTML are under
`/huyang2/double-loop/runs/p-fs3-003-comparison-20260806T234200Z-8372387`.
Their SHA256 values are
`e4b80fb3287d18e36769c05bb258d34743f9a2466e81ddbe3fc352fff34a5aaf`
and `3d0d97012ae7b2609cb5c087659e46af6a3c8ac0e85bdeb64157c8714ebd9c56`.
The manifest validates the JSON, HTML, and archived builder.

## 7. Formal Results

### 7.1 Registered Decision Gates

Activation requires mean absolute row gain at least `1e-4`, nonzero row and
board variation, and finite nonzero residual relative RMS. Primary quality
requires hard official51-64 macro loop5 exact `+0.02`, with no hard-range
blank regression over `0.01`. The alternate route requires mixed exact
`+0.03`, non-regressive official61-64 exact, and stronger same-board loop3-to5
correction. Fresh-process elapsed and peak-allocation overhead must both remain
below 10 percent. Integrity, activation, quality, and cost are conjunctive.

### 7.2 Optimization And Activation

Control/candidate train CE is `0.858617/0.854757`. Parameter count changes
`11,485,760 -> 11,485,795`, exactly the registered 35 parameters.

The router is clearly active at loop5:

- mean absolute row gain: `0.014254`;
- within-state row-gain standard deviation: `0.003989`;
- between-board gain standard deviation: `0.001069`;
- producer-update relative RMS: `0.898864`;
- residual relative RMS: `0.012122`;
- residual between-board standard deviation: `0.000421`.

Activation passes by a wide margin. This is stronger content modulation than
the single-payload codec (`0.002382` residual relative RMS), so a zero or dead
route cannot explain the quality result.

### 7.3 Mixed And Official Quality

Mixed exact is identical across all five loops. Control/candidate loop5 exact
is `0.025391/0.025391`; mixed loop5 blank changes
`0.545610 -> 0.542324`.

Official 512-board metrics are:

| Range | Arm | exact loops1-5 | blank loops1-5 |
| --- | --- | --- | --- |
| 51-55 | control | 0/0/0.001953/0.001953/0.001953 | 0.532345/0.566677/0.573945/0.573623/0.573766 |
| 51-55 | address-local | 0/0/0.001953/0.001953/0.001953 | 0.533920/0.566463/0.572728/0.572513/0.572871 |
| 56-60 | control | 0/0/0/0/0 | 0.473182/0.498061/0.501699/0.503140/0.503861 |
| 56-60 | address-local | 0/0/0/0/0 | 0.473937/0.494458/0.500635/0.501733/0.501973 |
| 61-64 | control | 0/0/0/0/0 | 0.504319/0.578523/0.589207/0.591954/0.591923 |
| 61-64 | address-local | 0/0/0/0/0 | 0.503495/0.578706/0.593846/0.596746/0.597387 |

Hard51-64 macro loop5 exact is unchanged
`0.000651 -> 0.000651`; mixed exact delta is also zero. Loop5 blank deltas for
51-55/56-60/61-64 are `-0.000895/-0.001888/+0.005464`. Preserving address
rows creates a real hardest-tail blank gain, but it does not convert any hard
range into more full-board solutions. Neither quality route passes.

### 7.4 Same-Board Loop Dynamics

All 256 case IDs, labels, and data hashes match independently in each hard
range. Mean wrong-cell trajectories are:

| Range | control loops1-5 | address-local loops1-5 | control L3-to5 | candidate L3-to5 | candidate L5 better/equal/worse |
| --- | --- | --- | ---: | ---: | --- |
| 51-55 | 25.74/24.35/23.92/23.86/23.93 | 25.65/24.25/23.98/23.87/23.88 | -0.016 | 0.109 | 102/57/97 |
| 56-60 | 29.70/28.20/28.08/28.04/28.02 | 29.76/28.40/28.04/28.02/27.96 | 0.066 | 0.086 | 98/69/89 |
| 61-64 | 31.86/27.20/26.61/26.51/26.43 | 31.65/27.21/26.23/26.08/26.06 | 0.184 | 0.168 | 121/57/78 |

The candidate finishes with fewer mean wrong cells in all three ranges, and
its 61-64 gain is broad: 121 boards improve versus 78 regress. However, its
loop3-to5 correction is slightly weaker on 61-64 and the registered alternate
route also requires mixed exact `+0.03`; the mixed delta is zero. The paired
dynamics are useful mechanism evidence, not a gate pass.

### 7.5 Cost

Fresh-process continuation time is `825.97s -> 955.36s`, so effective
throughput falls `15.497 -> 13.398` boards/s and elapsed overhead is `+15.66%`.
Peak allocated memory is `13,186.4 -> 13,958.9 MiB` (`+5.86%`); peak reserved
memory is `14,288 -> 15,060 MiB` (`+5.40%`). Memory passes, but elapsed cost
misses the registered 10-percent ceiling.

## 8. Decision

P-FS3-003 is discarded. Identity, gradient, activation, official-kernel,
exact-resume, data, source, and single-GPU integrity gates pass. Both exact
quality routes fail and elapsed overhead exceeds 10 percent. Do not rescue it
with row features, hidden width, bias, gain scale, normalization,
layer/head-specific parameters, extra steps, seed, LR, loss, batch, model
width/depth, payload count, or a nearby address variant.

The result closes a specific boundary: preserving address-local producer
updates is better than compressing them to one payload for hardest-tail blank
accuracy, but FutureSeed-side content modulation still does not create new
global solutions under matched compute. The next high-information move should
change the generic GDN recurrent memory/state update itself, rather than add a
fourth FutureSeed residual/router variant.

## 9. Submission Record

Not applicable. This is a mechanism experiment, not a platform submission.
