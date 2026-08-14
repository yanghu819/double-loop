# P-GDN3-042: Midpoint-Coherent Q/K

## 1. Metainfo

- Status: completed; discarded by registered quality and warmed-cost gates
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
compilation, shell parsing and `git diff --check` passed; no CPU model smoke was
run. R1 source `63493645` stopped before model construction because the
checker asked `inspect` for a Torch-Dynamo-wrapped function and received
`torch/_dynamo/eval_frame.py`; its non-science abort is retained. R2 changed
only provenance inspection to verify the exported function identity, the
`fla.ops.gdn2.chunk` module path and the unchanged full ops-tree hash.

R2 exact pushed/read-back source is
`39386276cc89c3e931579ae190254f242a47354d`. Its strict CUDA contract passes:
the candidate adds exactly 256 parameters, retains 4,096 state values and one
scan per layer, is bit-exact to the parent for zero and nonzero incoming state,
keeps both official `ChunkGDN2FunctionBackward` paths and native gradients,
and gives finite nonzero gradients to all eight new adapter heads. Opened
adapters preserve the midpoint and registered differential map within
`4.77e-7`, are head-permutation equivariant, and alter both output and state.

The fixed endpoint result is:

| metric | native control | midpoint-coherent Q/K |
| --- | ---: | ---: |
| balanced accuracy | 0.174750 | 0.057250 |
| future accuracy / CE / exact | 0.1610 / 2.91777 / 0.002 | 0.1000 / 3.71948 / 0.007 |
| past accuracy / CE / exact | 0.1885 / 2.90015 / 0.002 | 0.0145 / 5.09822 / 0.001 |
| joint exact | 0.000000 | 0.000000 |
| total query errors | 3,301 | 3,771 |
| wrong-key valid-value swaps | 770 | 331 |
| wrong-key fraction of errors | 0.233263 | 0.087775 |

The failure is not dead activation. Layer adapter RMS is `0.03885/0.01712`,
all eight heads are active, Q/K relative changes span `0.00835-0.02068`, and
alpha spans `0.95703-1.05469`. Terminal state remains finite and variable, and
native FutureSeed remains active with gate `0.52463`. The lower conditional
swap fraction is invalid as a quality win because total errors increase by
470 and both directional accuracies regress, with the past direction nearly
collapsing.

Elapsed, post-warm wall and allocation ratios are
`0.7288x/0.5352x/1.0160x`, but the independent warmed-step ratio is `1.6108x`
and fails the registered `1.20x` ceiling. Quality already fails every absolute
and relative gate except the conditional swap fraction.

Discard P-GDN3-042. P031/P036 already show that splitting erase from the
coherent key destroys row ownership; P042 now shows that directly contracting
or expanding native Q/K disagreement also destroys directional binding. The
positive P020 boundary should therefore be interpreted as better geometry
inside each full native address map, not evidence that the maps should be made
similar. Preserve the full native Q/K differential and exact coherent
read/write/erase ownership in successors. Do not rescue alpha range,
granularity, sharing, initialization, projection, normalization, regularize
coherence, or extend training. No Sudoku transfer is authorized.

Run:
`p-gdn3-042-qk-coherence-r2-l1024-20260814T205400Z-3938627`.
Contract/score/comparison/formal-log SHA256 are
`2ae3c35b98171d6e23a47695c1a5c2543aee7c0c1eb0eab9707b2f4e0735b893` /
`02ecf1c604aeb6164e02060d8f8d8951fcbbdbc6e640dc8dca06822bfbf820e6` /
`02ecf1c604aeb6164e02060d8f8d8951fcbbdbc6e640dc8dca06822bfbf820e6` /
`485580f50f3e7b5588b7cd280c2cc496061fef9a3bc8e4fe1dfffdd19349e0dc`.
Control/candidate checkpoint SHA256 are
`4d96b91f9278d38bd92e6f6cf35559ad067a8d5f3ea96d5cd519c558b8a346aa` /
`2c0c70a54a5ff7cbf4848541a1d3d1a625add0ab711f54d30b44413a8b281ee7`.
