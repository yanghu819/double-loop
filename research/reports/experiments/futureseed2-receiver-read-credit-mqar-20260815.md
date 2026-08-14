# P-FS2-011: Receiver-Native Terminal-Read Credit

## 1. Metainfo

- Status: preregistered; implementation complete; GPU contract pending
- Task: directional MQAR, sequence length 1024, four future and four past queries
- Carrier: D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed
- Fixed endpoint: 10 epochs, batch32, seed123, contemporaneous control then candidate
- External decoupled-key source reviewed: `yanghu819/GDN_decouple_k` at
  `c7667fd11d95d3d147f59bb3d4492989909b69ae`

## 2. Mechanism Hypothesis

The direct decoupled-key idea and increasingly constrained refinements are now
bounded. P031 split write/read and erase keys exactly and collapsed retrieval;
P036 anchored the erase key close to write ownership and also collapsed;
P042 preserved a midpoint while shrinking Q/K disagreement; P043 preserved
the raw Q/K bilinear pairing through a dual gauge; P044 transported live state
through a bounded token-varying frame. All were active, and none improved
absolute binding quality.

Those failures do not test whether native FutureSeed receives a useful
gradient for *readability in the receiver's own learned basis*. P-FS2-006
proposed receiver-native inherited-versus-live evidence, but its hand FP32
recurrence replay exceeded a preregistered parity tolerance before any quality
decision. The exact official graph still exposes the producer terminal state,
the inherited seed, and the receiver terminal state without replay.

Prediction: directly crediting the producer for making its inherited state
agree with evidence that the receiver eventually stores under the receiver's
own queries will reduce long-range wrong-key binding errors. Because the
receiver query and teacher are detached, the intervention should not distort
the fragile Q/K geometry seen in P031-P044.

## 3. Exact Intervention

Inference is native FutureSeed with no additional operation. During training,
for each producer-to-receiver route:

```text
F       = native normalized/gated producer terminal state
S_T     = exact official receiver terminal state
q_t     = exact receiver Q projection + Triton short convolution
r_F,t   = normalize_RMS(q_t.detach()^T F)
r_T,t   = normalize_RMS(q_t.detach()^T S_T.detach())
L_credit = mean_t,h,v (r_F,t - r_T,t)^2
L         = L_CE + 0.25 * mean_routes(L_credit)
```

The loss uses every token, board, head and value channel. It receives no label,
query-position mask, task rule, selector, search or repair signal. Gradients
may reach the producer state-producing path and receiving FutureSeed gate only.
The teacher, receiver Q and receiver GDN2 parameters are detached from the
auxiliary path. Parameter count, recurrent state, official scan count and
inference graph remain unchanged.

## 4. Independent Data Contract

Use the frozen directional MQAR generator and cache already used for the
P020-P044 comparison family:

- train examples: 10,000;
- validation examples: 1,000;
- sequence length: 1024;
- key/value pairs: 4;
- seed: 123;
- fixed train/test content hashes checked by the CUDA contract;
- same matched initialization file and warmup batch for both arms.

No historical score is a control. The only quality reference is the
contemporaneous native FutureSeed arm executed first in the same endpoint.

## 5. Integrity Contract

Before formal training, require all of the following on the sole visible GPU1:

- exact target UUID and CUDA index0;
- clean detached worktree from a pushed and API-read-back GitHub SHA;
- pinned FLA source SHA and Python-tree hashes;
- exactly two native `GatedDeltaNet2` layers and Triton short convolutions;
- exactly 661,584 parameters in both models and 4,096 recurrent values/layer;
- bit-exact eval and train logits against native FutureSeed at initialization;
- bit-exact nonzero-incoming-state outputs and terminal states;
- auxiliary graph contains exactly one producer `ChunkGDN2FunctionBackward`;
- combined CE/credit graph contains exactly two official backward paths;
- finite nonzero producer K/V/decay/erase/write and receiving FutureSeed-gate
  gradients;
- exactly zero auxiliary gradient in every receiving GDN2 parameter;
- finite nonzero all-token credit, both reads and board-varying receiver Q;
- no fallback, added parameter, state, inference scan or target use.

Any miss is an integrity failure. It does not authorize a tolerance, detach,
normalization or implementation rescue.

## 6. Fixed Falsifiers And Gate

Activation must show one active receiver route, finite unweighted credit in
`[1e-4, 4]`, weighted credit at least `1e-5`, nonzero inherited/live reads,
board-varying receiver queries and active native FutureSeed.

All quality checks are conjunctive:

- balanced accuracy `>=0.55` and `>=control+0.10`;
- future accuracy `>=0.50` and `>=control+0.08`;
- past accuracy `>=0.50` and `>=control+0.08`;
- joint exact `>=0.06` and `>=control+0.04`;
- fewer total query errors;
- wrong-key valid-value swap fraction reduced by at least `0.05`.

Any miss closes this exact training signal. Do not rescue auxiliary weight,
teacher target, detach boundary, normalization, token/layer mask, seed, data,
loss, width, depth or duration.

## 7. Cost And Scaling

The candidate adds no inference compute or memory. Training recomputes the
receiver Q projection/Triton convolution and two dense state reads. Fixed cost
ceilings against the contemporaneous control are:

- end-to-end elapsed `<1.30x`;
- post-warm wall `<1.30x`;
- independent warmed training step `<1.30x`;
- peak allocated CUDA memory `<1.12x`.

These ceilings are fixed before results and cannot be relaxed afterward.

## 8. Next Decision

If every contract, activation, quality and cost check passes, authorize one
matched hard-Sudoku transfer using the same training-only signal and native
inference architecture. If any check fails, close receiver terminal-read
distillation and stop nearby loss variants. The next mechanism must then alter
a genuinely different scalable state organization or training regime, not Q/K
coordinates, producer readout, replay cache or another auxiliary proxy.

## 9. Submission Record

Not applicable. This is a local architecture science gate.
