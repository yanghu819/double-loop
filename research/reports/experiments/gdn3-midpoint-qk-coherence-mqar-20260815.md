# P-GDN3-042: Midpoint-Coherent Q/K

## 1. Metainfo

- Status: preregistered; implementation and static checks complete; GPU gate pending
- Date: 2026-08-15
- Benchmark: directional MQAR L1024 wrong-key binding regime
- Fixed setting: D128/L2/H4/K32/V32, native FutureSeed, 10 epochs,
  batch32, seed123
- Arms: contemporaneous native GDN2 control, then one matched candidate
- Resource: one task-mode GPU, no concurrent model or evaluation process

## 2. Evidence And Hypothesis

P-GDN3-020 is the strongest clean main-state signal: an independent bounded
Log-SPD metric raises same-runtime balanced accuracy from `0.1735` to
`0.48225`. P-GDN3-025 independently shows that learned address organization
beats a fixed same-byte correction basis by `+0.343` balanced. P-FS2-007 shows
that exact-surprise replay preserves values, but `99.85%` of its remaining
errors bind a valid value to the wrong key. Address binding, not missing value
capacity, is therefore the unresolved mechanism.

The failed decoupled-key family P-GDN3-031/036 split erase from write and broke
row ownership. P-GDN3-041 injected the current query only into erase content
and also collapsed retrieval. Neither tests whether the independently learned
native Q and K maps themselves drift apart. The candidate tests the opposite
of key decoupling: preserve one coherent read/erase/write state while learning
how much of the Q/K differential is useful in each head coordinate.

## 3. Mechanism

After the unchanged official Q/K projections and Triton short convolutions,
form the midpoint and differential

`c=(q+k)/2`, `d=(q-k)/2`.

Each layer learns one H4xK32 tensor `a`, with

`alpha = 1 + 0.5*tanh(a)`,
`q' = q + (alpha-1)*d`,
`k' = k - (alpha-1)*d`.

Thus `q'=c+alpha*d` and `k'=c-alpha*d`. At `a=0`, both tensors are exactly the
parent Q/K tensors because the residual multiplier is exactly zero. The
midpoint is preserved for every token/head/coordinate, while `alpha` remains
in `[0.5,1.5]`: values below one contract read/write disagreement and values
above one retain the option to specialize. The unchanged pinned-official
`chunk_gdn2` performs its normal Q/K normalization and sole recurrent scan.

Across two layers the candidate adds exactly 256 parameters and no recurrent
state, token, scan, cache, custom kernel, task rule, selector, or auxiliary
loss. Native FutureSeed still transfers the complete terminal K32xV32 state.

## 4. Novel Boundary

This is not P020 metric scaling: a shared SPD factor changes within-map
geometry while preserving the Q/K differential. It is not cross-layer metric
sharing or state pullback. It does not split erase/write/read keys, add query
feedback to the erase equation, add capacity, replay events, route states, or
rewrite values. It tests a previously open boundary: bounded coherence between
the native read and write address maps while retaining their full learned
projections and the original coherent state transition.

## 5. Strict CUDA Contract

The exact pushed SHA in a clean detached worktree on the sole GPU must prove:

1. exact CUDA index0/UUID, pinned FLA and Zoology SHAs, Triton short
   convolutions, two official `ChunkGDN2FunctionBackward` paths, and no
   fallback;
2. exact matched parent tensors and exact zero-adapter output and terminal
   states for both zero and nonzero incoming recurrent state;
3. exact parameter delta 256, state size 4,096/layer, one scan/layer, finite
   nonzero adapter gradients in both layers, and unchanged native gradients;
4. opened adapters preserve the Q/K midpoint, scale their differential by the
   registered alpha, remain head-permutation equivariant, and keep alpha in
   `[0.5,1.5]`;
5. exact train/test hashes and no concurrent GPU model process.

Any miss closes the implementation before training. No CPU model smoke is
allowed.

## 6. Fixed Matched Protocol

Construct one canonical native initialization, save it, run the native control,
then map the same parent tensors into the candidate and run it. Both arms use
the same source, data, optimizer, schedule, batch order, seed and evaluation
cases. The only candidate-only tensors are zero-initialized H4xK32 adapters.
There is no repeated control, seed, LR, loss, width, duration or run-order arm.

## 7. Registered Gates

Activation requires both layers and all eight heads active, adapter RMS and
Q/K differential change at least `1e-3`, finite coordinate variation, alpha
inside `[0.5,1.5]`, exact midpoint error at most `1e-5`, and terminal state RMS
finite in `[1e-4,1e4]`. Native FutureSeed must remain active.

Quality requires every condition:

- balanced accuracy at least `0.55` and at least control `+0.10`;
- future and past accuracy each at least `0.52` and control `+0.07`;
- joint exact at least `0.08` and control `+0.04`;
- fewer total query errors; and
- wrong-key valid-value swap fraction at least `0.05` below control.

The historical `0.7475` endpoint is retained as context but is not a gate,
because its runtime carrier has been retired. A pass authorizes one strict
Sudoku transfer only after explaining the MQAR mechanism evidence.

Elapsed, post-warm wall and independently warmed-step ratios must each be below
`1.20x`; peak allocation must be below `1.10x`. Any integrity, activation,
quality or cost miss discards midpoint Q/K coherence. There is no alpha cap,
scalar-vs-coordinate, sharing, initialization, projection, normalization,
seed, LR, loss, batch, width, depth, epoch or duration rescue.

## 8. Required Artifacts

Archive both scores/cases/checkpoints, wrong-key summaries, per-layer alpha and
Q/K geometry, FutureSeed diagnostics, timing/memory, source/config/log hashes,
GPU provenance and one comparison JSON under `/huyang2/double-loop/runs`.

## 9. Decision

Implementation, endpoint, strict checker and sole launcher are complete. Python
compilation, shell parsing and `git diff --check` pass; no CPU model smoke was
run. Pending exact pushed SHA, clean detached worktree, strict GPU contract and
the sole fixed endpoint.
