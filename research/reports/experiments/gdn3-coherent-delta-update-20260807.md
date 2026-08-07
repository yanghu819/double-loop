# P-GDN3-005: Coherent Delta Erase/Write Update

## 1. Metainfo

- Status: approved; implementation and contracts pending
- Date: 2026-08-07
- Branch: `codex/gdn3-coherent-delta-20260807`
- Benchmark: official/full-diversity hard 9x9 Sudoku
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent mechanism: D256/L12 position-QK GDN3 plus native FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen matched control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

The formal source SHA will be frozen only after this report, implementation,
launcher, and contract are pushed. The AIStation source must be a clean
detached worktree built from that pushed SHA.

## 2. Mechanism Boundary

P-FS3-001 through P-FS3-003 all learned active FutureSeed content residuals.
The address-local variant even preserved every KxV payload and improved the
61-64 blank tail, yet none changed hard macro or mixed exact, and all missed
the cost gate. This closes nearby transfer-side routing as the next step.

P-GDN3-004 supplies the complementary clue. Stable position-QK addresses
create genuine late-loop correction and nonzero hard exact, but the run stalls
below its step3000 opening gate. The remaining failure can lie inside the
generic recurrent state edit rather than in how a terminal state is passed to
the next layer.

## 3. Hypothesis

The official GDN2 layer predicts an erase gate `b` over K coordinates and a
write gate `w` over V coordinates independently. The underlying delta update,
however, is one state-edit event: erase the old value addressed by K, then
write the new V target. If their aggregate strengths drift apart, the update
can under-erase stale state or under-write its replacement. This predicts slow
soft correction without enough stable board-level fixed points to open exact.

The falsifiable prediction is that a small learned positive coupling should
reduce the erase/write gate gap and improve hard exact or mixed exact. If the
coupling remains inactive, learns a negative mean, fails to contract the gap,
or only changes soft accuracy, the hypothesis fails.

## 4. Candidate

For every token and head, after the official projections but before the
unchanged official kernel, define:

```text
b0 = sigmoid(b_raw)
w0 = sigmoid(w_raw)
c  = 0.5 * (mean_K(b0) + mean_V(w0))
m  = tanh(a[layer, head])
b1 = clamp(b0 + m * (c - b0), 0, 1)
w1 = clamp(w0 + m * (c - w0), 0, 1)
```

Each of 12 layers has eight scalar `a` parameters, for exactly 96 added
parameters. They are zero initialized and excluded from weight decay. At
`m=0`, every gate, output, and terminal recurrent state must be bit-exact to
the parent, with and without an incoming recurrent state. The candidate keeps
the pinned official `chunk_gdn2`, position-QK addressing, full KxV state,
native terminal FutureSeed, one forward traversal, and linear complexity.

This is not gate-temperature tuning. There is no fixed scale, auxiliary loss,
Sudoku rule, search, repair, extra scan, new state slot, or second seed.

## 5. Frozen Configuration

- D256/L12/H8/K32/V32, channel multiplier4
- 12 pinned official-FLA `GatedDeltaNet2` layers and Triton short convolution
- position-QK addressing, random traversal
- native terminal FutureSeed, loop5, all-loop CE
- official full-diversity curriculum and exact step3000 parent
- BF16, effective batch128, optimizer/RNG/data order exact resume
- LR0.0015, weight decay0.001, seed52
- candidate-only step3000-to3100 continuation

The frozen control is the existing exact terminal continuation to step3100;
it is not rerun.

## 6. Launch Gates

The strict CUDA contract must prove:

1. exactly CUDA index0 and the registered UUID;
2. pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. 12 exact official `GatedDeltaNet2` layers and
   `ChunkGDN2FunctionBackward`;
4. exactly 96 inserted parameters and the exact migration set;
5. bit-exact full output and all 12 terminal states at zero initialization;
6. bit-exact per-layer output and terminal state with nonzero incoming state;
7. finite nonzero gradient for every layer's coupling parameter;
8. gate range `[0,1]` and strict gap contraction under a positive opened mix.

Only after the contract passes may a single exact step3000-to3001 full-stack
probe run. It must accept the declared migration, preserve optimizer/RNG/data
order, save a complete checkpoint and metrics JSON, use the official CUDA
path, and show a finite opened coupling after the optimizer step. Any contract
or integrity miss blocks the formal run.

## 7. Science And Cost Gates

Activation at step3100 must satisfy all of:

- mean coupling `m > 0`;
- mean absolute coupling `>= 1e-4`;
- finite nonzero coupling gradients;
- post/pre erase-write gap RMS `< 1`.

Quality passes by exactly one of:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with each hard
   blank range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

Elapsed overhead and peak allocated memory must each remain below `10%` versus
the frozen control. Timing must use the existing independent warmup protocol.
Any miss discards the mechanism without coupling scale, seed, LR, loss, batch,
width, depth, duration, or nearby update rescue.

## 8. Commands

Strict contract:

```bash
CUDA_VISIBLE_DEVICES=0 \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
TRITON_CACHE_DIR=/huyang2/double-loop/.cache/triton \
TORCHINDUCTOR_CACHE_DIR=/huyang2/double-loop/.cache/torchinductor \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
TMPDIR=/huyang2/double-loop/.cache/tmp \
FLA_DISABLE_BACKEND_DISPATCH=1 FLA_CONV_BACKEND=triton \
/opt/conda/bin/python \
experiments/rwkv_fs_sudoku/check_gdn3_coherent_delta_cuda.py
```

Exact full-stack probe:

```bash
CUDA_VISIBLE_DEVICES=0 GDN3_FULL_STACK_PROBE=1 \
scripts/run_gdn3_coherent_delta_arm.sh
```

The formal command is the same launcher without
`GDN3_FULL_STACK_PROBE=1`.

## 9. Decision

Pending. No GPU launch is authorized until the implementation, report, and
contract are pushed, a clean detached worktree exists, the parent hash is
verified, the sole GPU is idle, and both launch gates pass.
