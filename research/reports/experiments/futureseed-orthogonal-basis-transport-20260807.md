# P-FS3-004: Orthogonal State-Basis Transport

## 1. Metainfo

- Status: approved, not launched
- Date: 2026-08-07
- Branch: `codex/fs3-orthogonal-basis-transport-20260807`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent: D256/L12/H8/K32/V32 position-QK GDN3 plus native terminal
  FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen control: `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Evidence Boundary

Native FutureSeed transfers producer terminal KxV state directly into the next
layer, but the producer and receiver own independently learned recurrent K and
V coordinates. The existing scalar head gate and unit-RMS normalization can
control transfer strength and magnitude, but cannot align those coordinate
bases.

P-FS3-001/002/003 changed the transferred content, update magnitude, or
address-local residual. P-FS2-005 learned a downstream hidden-state readout.
P-GDN3-009/P-GDN3-011 rotated state or payload inside one recurrent layer.
None learned an invertible coordinate map on the actual cross-layer
FutureSeed edge. P-GDN3-014 instead changed physical state width and failed
parent identity. P-FS3-004 keeps the physical K32xV32 state unchanged.

## 3. Mechanism

For every adjacent producer-to-receiver edge and head, learn independent
skew-symmetric generators `A_K` and `A_V`. Convert them to orthogonal maps with
the Cayley transform:

`R(A) = (I - A/2)^-1 (I + A/2)`

and transport the producer state as:

`S_receiver = R_K S_producer R_V^T`.

The implementation applies `(R-I)` residuals around the original tensor so
zero generators preserve the parent state bit-exactly while retaining a direct
first-order gradient. Orthogonal transport preserves Frobenius geometry and
does not add state, tokens, scans, recurrent cores, task logic, or a reverse
pass.

Each K32 or V32 generator has `32*31/2 = 496` packed parameters. Across 11
edges and 8 heads the exact parameter delta is
`11*8*(496+496) = 87,296`; persistent recurrent-state delta is zero.

## 4. Falsifiable Prediction

If cross-layer coordinate mismatch limits FutureSeed, one exact resumed step
should activate both K and V rotations on all 11 edges, with nonzero board and
head variation while preserving state norm. A 100-step continuation should
then improve hard-board closure or mixed exactness without degrading 61-64
late-loop correction. If rotations activate but exact remains flat, basis
misalignment is not the current bottleneck. If the production path violates
identity, orthogonality, bounded state geometry, or direct-gradient access,
the implementation claim is false before any science run.

## 5. Strict CUDA Contract

Before any continuation, exact pushed source in a clean detached worktree must
prove all of:

1. CUDA index0 and the registered UUID are the only visible GPU and compute app;
2. pinned FLA source SHA is `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. all 12 recurrent layers are official `GatedDeltaNet2`, with
   `ChunkGDN2FunctionBackward` in every terminal-state graph and no fallback;
4. zero-init full output and all 12 terminal states are bit-exact to the frozen
   parent function;
5. every edge preserves arbitrary finite nonzero incoming state bit-exactly at
   zero initialization;
6. parameter delta is exactly 87,296 and state/token/scan/core deltas are zero;
7. every one of the 11x8 K and V generators receives a finite nonzero direct
   first-order task gradient;
8. opened K/V maps change full-model output, retain finite states, and keep
   every layer's terminal RMS within `4x` its matched control;
9. FP32 orthogonality and norm max error are each `<=3e-5`, BF16 storage norm
   max error is `<=5e-3`, and head permutation error is `<=3e-6`;
10. no NaN, OOM, source/data drift, CPU model path, concurrent GPU process,
    task dependency, or silent fallback occurs.

## 6. Step3001 Production Gate

The exact-resume one-step probe is migration and production-fit evidence only.
It must preserve optimizer/RNG/data order and write complete source/config/log,
metrics, and checkpoint hashes. All 11 edges must have K rotation, V rotation,
and transported-state residual relative RMS `>=1e-4`; board and head variation
must be finite and nonzero. Production FP32 norm/orthogonality max error must be
`<=3e-5`, BF16 storage norm error `<=5e-3`, and terminal RMS `<=4x` control.
Any miss closes P-FS3-004 before formal training. The probe score is not a
science result.

## 7. Matched Science And Cost Gates

Only after contract and production probe pass may one candidate-only exact
step3000->3100 continuation run. Do not repeat the frozen control.

- Primary: hard51-64 macro loop5 exact improves by at least `+0.02`, and each
  official hard range blank accuracy regresses by no more than `0.01`.
- Alternate: mixed loop5 exact improves by at least `+0.03`, 61-64 does not
  regress, and same-board loop3->5 wrong-cell correction is stronger.
- Stability: every rotation/state remains finite, FP32 geometry stays within
  `3e-5`, BF16 norm error within `5e-3`, and terminal RMS within `4x` control.
- Cost: independent warmed elapsed overhead `<35%` and peak allocated-memory
  overhead `<15%` versus the frozen control.

Any activation, stability, quality, integrity, or cost miss discards P-FS3-004.
There is no rank, sharing, axis, map, angle, normalization, precision, seed,
LR, loss, batch, width/depth, or duration rescue.

## 8. Results

Initial pushed SHA `6f40c0a2ac7e3a9a055fa5738be88970538db69a`
reached and passed the registered identity, nonzero-state, parameter,
direct-gradient, synthetic geometry, opened full-model geometry, and official
backward assertions. It then exited status1 in the final provenance-summary
call because the checker passed the `FutureSeedRWKV` reasoner directly to a
helper that requires an outer object with `.reasoner`. This is a checker
object-shape error after the mechanism assertions, not a model, data,
integrity, or science failure. GPU1 released naturally to 0 MiB.

R1 log SHA256 is
`a8e31b690fdf960acee0366142f8496b29ccbeb1282b8640b42ed61b09707893`;
its non-science abort JSON SHA256 is
`a4d7083c114276e037d03c8b1b272945e41a1b876e5e3a6b9a81e76b6e36a454`.
The correction only supplies the required outer runtime container; it does not
change mechanism code, parameters, thresholds, data, or training.

Corrected pushed SHA `f5beffffcde7df9c38bc570a40caf17fc0fe52d9`
passes strict R2 status0 on the only visible GPU1. All 12 official GDN2 layers
and backward paths are present; zero-init output and all 12 states are exact;
the parameter delta is 87,296; minimum edge/head direct-gradient maxima are
`4.11112e-4` for K and `2.39802e-4` for V. In the opened full model, minimum
K/V rotation relative RMS is `0.054892/0.054574`, transported-state residual
relative RMS is `0.078920` with minimum `0.076569`, and maximum terminal-state
RMS ratio is `1.003604`. FP32 norm error is `1.78814e-7`, orthogonality error
is `1.13249e-6`, BF16 storage norm error is `1.78814e-7`, and head permutation
error is zero. Contract log SHA256 is
`5fec22ed10fe0000cf46765a8f3ee85a81aa9bc535b97a964ce6d4c470978101`.
GPU1 released naturally to 0 MiB.

The first exact-resume probe attempt exited before any training step because
the generic checkpoint loader's explicit content-upgrade allowlist omitted the
two new parameters, while reporting exactly
`future_seed_basis_transport.row_angles/col_angles` as missing and no
unexpected tensors. No checkpoint or score was produced. This is a migration-
wiring error rather than parent-state drift: the semantic upgrade flag,
subsequent exact missing-set assertion, and optimizer-state expansion were
already present. Probe R1 log/abort SHA256 are
`c75d622007108525c54913926bc221e98acf162f0cbfa51f6f7be96dc36fff3d` and
`5d316768a25d15bea224900cb475401b88137679172371696bda3ca36d67a7f4`.
The only correction adds those two exact names to the allowlist; it does not
change the model, initialization, optimizer, data, threshold, or run command.
A new pushed SHA must repeat strict CUDA contract before a corrected probe.
No benchmark score exists yet.

## 9. Decision

Strict R2 on `f5befff` passes, but its first probe attempt did not reach a
training step because of the explicit loader allowlist omission. Permit only
the exact two-name migration-wiring correction, followed by a new pushed-SHA
contract and corrected step3001 probe. Formal continuation remains blocked.
