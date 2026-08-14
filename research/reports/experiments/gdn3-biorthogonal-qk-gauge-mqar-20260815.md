# P-GDN3-043: Biorthogonal Q/K Gauge

## 1. Metainfo

- Status: preregistered; implementation complete; CUDA contract pending
- Date: 2026-08-15
- Benchmark: directional MQAR L1024 wrong-key binding regime
- Fixed setting: D128/L2/H4/K32/V32, native FutureSeed, 10 epochs,
  batch32, seed123
- Arms: contemporaneous native GDN2 control, then one matched candidate
- Resource: one task-mode GPU, no concurrent model or evaluation process

## 2. Evidence And Hypothesis

The clean-room test inspired by `GDN_decouple_k` is already bounded on both
sides. P031/P036 show that giving erase a separate key destroys coherent row
ownership even when the correction is tangent-anchored and key cosine remains
about `0.91`. P042 tests the opposite intervention and also fails: directly
contracting or expanding native Q/K disagreement lowers balanced accuracy from
`0.17475` to `0.05725`. Read, write and erase must retain one coherent address
owner, and the full native Q/K differential must remain available.

P020 nevertheless shows a large positive address-geometry signal: an
independent per-layer bounded Log-SPD map raises balanced accuracy from
`0.1735` to `0.48225`. The remaining open question is whether GDN2 needs a
stable change of address coordinates rather than either a split key or forced
Q/K similarity. The candidate predicts that inverse-transpose dual coordinates
can calibrate normalized read/write geometry while exactly preserving the
unnormalized native pairing algebra.

## 3. Mechanism

Each layer/head learns a symmetric trace-free generator `G` with bounded
Frobenius norm. Define

`C = exp(0.5*log(2)*G)`.

After the unchanged native Q/K projections and Triton short convolutions,
apply

`q' = q C^-T`, `k' = k C`.

Because `G` and `C` are symmetric, the implementation uses
`q'=q exp(-sG)` and `k'=k exp(sG)`. Before the official kernel's unchanged
L2 normalization,

`q' k'^T = q C^-T C^T k^T = q k^T`.

This retains every native Q/K distinction and the same coherent K address for
erase and write, while changing the coordinate geometry seen after
normalization and in the KxV state rows. The trace-free bounded generator keeps
the map volume-preserving, with factor eigenvalues inside approximately
`[2^-0.5, 2^0.5]` and condition number at most two.

The two-layer candidate adds exactly 4,216 parameters, no persistent state,
token, scan, cache, second core, custom recurrence, task rule, selector or
auxiliary loss. Native FutureSeed still transfers the complete terminal
K32xV32 state. Zero generator is parent-exact through an explicit residual
path.

## 4. Novel Boundary

This is not a rescue of P031/P036: there is no second erase key and no change
to read/write/erase ownership. It is not P042: the native Q/K differential is
untouched rather than rescaled. It is not P020 metric scaling: P020 multiplies
Q and K by the same factor and deliberately changes their bilinear metric;
P043 uses inverse-transpose factors and preserves the raw bilinear pairing.
It is a single causal test of dual-coordinate calibration suggested by the
decouple-key failure boundary.

## 5. Strict CUDA Contract

The exact pushed SHA in a clean detached worktree on the sole GPU must prove:

1. exact CUDA index0/UUID, pinned FLA and Zoology SHAs, Triton short
   convolutions, two official `ChunkGDN2FunctionBackward` paths and no
   fallback;
2. exact matched parent tensors and exact zero-gauge output and terminal states
   for both zero and nonzero incoming recurrent state;
3. exact parameter delta 4,216, state size 4,096/layer, one scan/layer, finite
   nonzero gauge gradients in every head and unchanged native gradients;
4. opened gauges preserve all-pair raw Q/K products within `1e-2` relative RMS
   at production BF16, satisfy applied inverse error at most `1e-2`, theoretical
   trace/logdet/inverse constraints, head permutation equivariance and the
   fixed factor bounds;
5. exact train/test hashes and no concurrent GPU model process.

Any miss closes the implementation before training. No CPU model smoke is
allowed.

## 6. Fixed Matched Protocol

Construct one canonical native initialization, save it, run the native control,
then map the same parent tensors into the candidate and run it. Both arms use
the same source, data, optimizer, schedule, batch order, seed and evaluation
cases. The only candidate-only tensors are zero-initialized per-layer gauges.
There is no repeated control, seed, LR, loss, width, duration or run-order arm.

## 7. Registered Gates

Activation requires both layers and all eight heads active, raw adapter RMS and
Q/K changes at least `1e-3`, head-level factor variation at least `1e-4`,
production replay exact, pairing and applied inverse errors at most `1e-2`,
trace/logdet constraints, factor eigenvalues in `[0.69,1.45]`, condition number
at most `2.10`, and finite variable terminal state. Native FutureSeed must
remain active.

Quality requires every condition:

- balanced accuracy at least `0.55` and at least control `+0.10`;
- future and past accuracy each at least `0.52` and control `+0.07`;
- joint exact at least `0.08` and control `+0.04`;
- fewer total query errors; and
- wrong-key valid-value swap fraction at least `0.05` below control.

Elapsed, post-warm wall and independently warmed-step ratios must each be below
`1.35x`; peak allocation must be below `1.12x`. The wider time ceiling is fixed
before results because the mechanism evaluates two 32x32 matrix exponentials
per layer. Any integrity, activation, quality or cost miss discards the complete
biorthogonal-gauge family. There is no cap, generator, symmetric-vs-general,
sharing, rank, initialization, projection, normalization, seed, LR, loss,
batch, width, depth, epoch or duration rescue.

## 8. Required Artifacts

Archive both scores/cases/checkpoints, wrong-key summaries, per-layer gauge,
factor, pairing and terminal-state geometry, FutureSeed diagnostics,
timing/memory, source/config/log hashes, GPU provenance and one comparison JSON
under `/huyang2/double-loop/runs`.

## 9. Decision

Pending strict CUDA contract and the single fixed endpoint. A full pass admits
one strict Sudoku transfer; any miss closes this family and returns mechanism
selection to a genuinely different live recurrent address/state update.
