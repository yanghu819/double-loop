# P-FS2-015 Receiver-Native Momentum Conv Prefill

## 1. Research Question

Does native Momentum FutureSeed lose useful short-range address ownership because
it transports terminal `[S,M]` but discards the producer's Q/K/V short-convolution
boundary?

## 2. Evidence And Hypothesis

P059 is the best verified carrier at directional MQAR L1024, reaching
balanced/future/past/joint `.94425/.95150/.93700/.82400`. Its remaining 223
errors contain 151 valid values assigned to adjacent write owners. P069 then
showed that removing Q/K width-four convolution destroys retrieval rather than
sharpening it: balanced accuracy falls to `.36275` and wrong-key swaps rise to
1,491. The short-convolution front end therefore forms useful local addresses.

The distinct FS2 hypothesis is that `[S,M]` carries global value and direction
but omits this learned local boundary. The final four producer hidden tokens are
causal summaries of the complete prefix. If the receiver reprojects those
tokens through its own Q/K/V basis and its own exact width-four Triton
convolutions, the resulting initial convolution state can restore local owner
continuity without transporting producer-basis keys or changing the recurrent
transition.

## 3. Fixed Mechanism

Keep P059's exact external Momentum DeltaNet SHA `c6e77fa`, both native chunk
scans, full Q/K/V projections and short convolutions, `[S,M]` recurrent state,
output correction and native terminal-state FutureSeed. After layer 0, retain
exactly its final four post-state-mixer hidden tokens. Layer 1 applies its own
Q/K/V projections and exact width-four Triton convolutions to that evidence and
uses the three resulting final convolution states as its initial conv cache.
Each stream/head has one zero-initialized `tanh` gate.

The candidate adds exactly 12 parameters, no persistent recurrent values and no
scan. It is not producer-basis K/V transport, top-k admission, sparse replay,
reverse scanning, a selector, a second memory or task-specific logic.

## 4. Strict CUDA Contract

The exact pushed SHA in a clean detached worktree must prove the registered
A800 and pinned source hashes, exact shared-parent loading, 599,684 parameters,
unchanged 8,192 `[S,M]` values per layer, two native Momentum chunk backwards,
six exact receiver Q/K/V width-four Triton convolution paths, bit-exact logits
and arbitrary nonzero-state output/state identity at zero gates, nonzero gate
gradient, ordered boundary dependence, evidence gradient and no fallback.

Any contract or provenance miss closes the implementation before training.

## 5. Prediction And Gates

Run one candidate-only D128/L2/H4/K32/V32 directional MQAR L1024 endpoint for
10 epochs, batch32, seed123 from the exact P059 matched initialization and
frozen data. Pass requires all of:

- one native `[S,M]` FutureSeed route and one active conv-prefill route;
- finite board-varying evidence/cache and mean absolute prefill gate at least
  `.001`;
- balanced and future accuracy each improve by at least `.005`;
- past accuracy regresses by at most `.003`;
- joint exact improves by at least `.005`;
- errors are at most 200 and wrong-key swaps at most 120;
- adjacent-owner swaps fall by at least 20% and conditional wrong-key share is
  at most `.60`;
- elapsed, post-warm and warmed-step ratios are below `1.15x`, and allocation
  below `1.05x`.

Any miss closes this complete four-token receiver-native conv-prefill mechanism.
There is no token-count, stream, gate, scale, projection, seed, LR, loss, batch,
width, depth or duration rescue. Only a full pass counts as a better FS2 and
authorizes transfer to hard Sudoku.

## 6. Result

The exact pushed/read-back source
`8705a098a709fcca6a62b8856a6e95a12a17db6a` ran from clean detached
worktree `/huyang2/double-loop/worktrees/p-fs2-015-8705a09` on A800
`GPU-c1d7c624-a393-befa-3807-7e00602d65ca`. The strict CUDA contract passed.
It proves exact +12 parameters, unchanged `[S,M]` state and scan count, two
native Momentum backwards, bit-exact zero-gate parent logits and arbitrary
nonzero-state output/state, finite gate/evidence gradients and ordered evidence
dependence `.074004/.039228` in output/state. Contract JSON SHA256 is
`be8ffe87f18c78ba711318056a301b65c62eadf86c7649ed8b37148daddd0ddf`.

The formal mechanism is fully active. The sole receiving gate reaches mean
absolute value `.051609`; evidence RMS/board std are `.345615/.006692`, raw
cache RMS is `.181327`, and mixed-cache RMS/board std are
`.015161/.000366`. Both layers keep bounded state and Momentum. Quality still
collapses:

| Metric | P059 control | P-FS2-015 | Delta |
|---|---:|---:|---:|
| balanced accuracy | .94425 | .35600 | -.58825 |
| future accuracy | .95150 | .36450 | -.58700 |
| past accuracy | .93700 | .34750 | -.58950 |
| joint exact | .82400 | .00100 | -.82300 |
| total errors | 223 | 2,576 | +2,353 |
| wrong-key valid-value swaps | 151 | 1,446 | +1,295 |
| adjacent-owner swaps | 151 | 1,424 | +1,273 |

Only 47 old swaps are repaired; 1,359 parent-correct queries become wrong-key
errors and 1,061 become other errors. Elapsed/post-warm/warmed/allocation ratios
are `1.76644/1.75676/.94571/1.00050x`. The formal process, including compile
and validation gaps, has active-sample mean/peak utilization `48.58%/75%` and
observed memory peak `2,284 MiB`.

Comparison, cases and checkpoint SHA256 are respectively
`613b7bfe59d0701e6ec9f8dedaa4858dfeee0ba4c058244ab23293422bb0c418`,
`10f102d5f1e730ea8f7e79de3ff5f664e14b96bd425c617a46052d12358d86d5`
and `89aecb98c23c76b8c66e54ad6d11dc4b032df4514e907973ec7d6a289d159713`.

## 7. Decision

Discard P-FS2-015. This is neither dead activation nor unstable recurrent
state. Receiver layer 1 already applies its Q/K/V projections and causal short
convolutions to the complete layer-0 hidden sequence at aligned positions.
Injecting layer 0's terminal hidden tail as layer 1's initial convolution cache
duplicates evidence and treats end-of-sequence representations as immediate
predecessors of token zero. That circular tail-to-head local geometry causes
broad ownership corruption. Close prefill token count, stream selection, gate,
projection, scale and every training rescue. Future FS2 work must improve the
semantics of the transported future state itself, not duplicate an aligned
feed-forward path through a misaligned convolution boundary.
