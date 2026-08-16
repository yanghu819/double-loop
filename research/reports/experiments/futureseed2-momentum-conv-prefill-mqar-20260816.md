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

Pending.

## 7. Decision

Pending the sole registered endpoint.
