# P-FS2-007: Receiver-Native Surprise Replay FutureSeed

## 1. Metainfo

- Status: preregistered; implementation ready for strict CUDA contract
- First decision field: directional MQAR L1024 wrong-key binding regime
- Compute: one AIStation A100-SXM4-80GB, CUDA index 0 only
- Model: D128/L2/H4/K32/V32 pinned official FLA GDN2
- Training: 10 epochs, batch32, seed123, 10,000/1,000 examples
- Arms, in fixed order: contemporaneous native FutureSeed baseline,
  recency-K16 replay, exact-surprise-K16 replay
- New parameters: zero
- Persistent recurrent-state delta: zero

## 2. Evidence And Hypothesis

The validated historical L1024 FutureSeed endpoint reached balanced accuracy
`0.7475` and joint exact `0.339`, but `80.69%` of its errors were correct values
bound to the wrong key. P020 showed that learned address geometry matters, yet
left `94.16%` wrong-key swaps. P025 showed a causal benefit from learned
organization of exact committed edits, but its producer-basis compressed bank
remained below P020. V-only state expansion, extra banks, write controllers,
producer codecs, terminal-state basis transport, and dense online inverse-Gram
state have already failed their registered gates.

The remaining hypothesis is narrower: terminal-state FutureSeed forces all
producer events through one producer-basis KxV summary. A bounded tape of the
largest actual committed edits may preserve the few events most vulnerable to
overwrite, while replaying canonical hidden/position evidence through the
receiver's own projections avoids cross-layer K/V basis mismatch.

This is not HYPIC/PIC serving reuse, a learned selector, producer-basis K/V
transport, a second correction bank, Sudoku logic, search, repair, or a Raven
write controller.

## 3. Mechanism

The producer's main pinned-official `chunk_gdn2` call is unchanged. The
existing official forward exposes the committed value residual already used
by the kernel:

```text
e_t = w_t * v_t - (b_t * k_t)^T [Diag(exp(g_t)) S_(t-1)]
```

The implementation captures that exact internal `e_t` without replaying the
recurrence and scores each token by `||e_t||_2`. The surprise arm selects the
top 16 tokens per board; the recency control selects the last 16. Both restore
original token order, retain the producer layer's full residual-stream hidden
vectors with their already-added positional evidence, and discard producer
K/V/state coordinates.

Before the next layer's normal L1024 scan, its own unchanged GDN2 mixer scans
the selected 16 hidden vectors once and uses that receiver-native terminal
state as FutureSeed. The normal receiver scan then runs unchanged. The event
tape is rebuilt at every layer and forward; it is not carried across a macro
loop. Admission has no trainable parameters, while gradients through selected
hidden evidence and the receiver replay remain end-to-end.

## 4. Why Existing Failures Do Not Cover It

- FS3 producer codec/router variants compressed producer-basis state content;
  this mechanism retains complete canonical token evidence and reprojects it.
- P-FS3-004 rotated a whole producer KxV state into the receiver basis; this
  mechanism never transports producer K/V coordinates.
- P025 wrote all detached committed edits into a compressed correction state;
  here exact edit is only an admission score and payload is receiver-native.
- P027 changed the live recurrence with a dense KxK inverse state; the main
  recurrence here remains pinned official and the complement is fixed K16.
- Recency-K16 has identical tape bytes and official replay cost, isolating the
  value of exact-surprise admission.

## 5. Strict CUDA Contract

Before training, the exact pushed SHA in a clean detached worktree must prove:

1. one visible A100 CUDA index0 with exact registered UUID;
2. pinned FLA SHA `9c8e42e...d85e`, exact GDN2 layer/ops hashes, Triton short
   convolution, disabled backend dispatch, and clean Zoology SHA `1ad20d1`;
3. exact historical train/test data hashes;
4. baseline, recency and surprise have identical parameter names, tensors,
   hashes and counts;
5. capture changes neither producer output, terminal state nor committed edit
   for zero or nonzero incoming state;
6. a synthetic event field makes surprise select positions `0..15` and
   recency select `48..63`, both restored in time order;
7. replay state changes when selected evidence order changes;
8. a full candidate backward contains exactly three
   `ChunkGDN2FunctionBackward` paths and finite nonzero gradients; and
9. no fallback, CPU model run or concurrent GPU process occurs.

Any miss closes this exact implementation before science. It does not license
a tolerance, capture, K, precision or training rescue.

## 6. Fixed Matched Protocol

All three arms use the same source, data hashes, initialization, parameter
count, optimizer, schedule, batch order, seed and evaluation cases. RNG is
reset before each arm. The historical source `77e5539` score and cases remain
SHA-locked as an absolute reference; the contemporaneous baseline controls
known runtime optimization drift and supplies the end-to-end cost denominator.

The historical baseline is evidence, not silently claimed reproducible. A
runtime baseline miss does not relax the absolute candidate bar.

## 7. Activation, Quality And Cost Gates

Both replay arms must have exact K16 admission, finite nonzero committed-edit
surprise, token variation, active receiver replay state, board variation and
exactly two main plus one replay official scan.

Surprise must satisfy every quality condition:

- balanced accuracy `>=0.85`;
- balanced accuracy at least `+0.10` over historical `0.7475`;
- balanced accuracy at least `+0.10` over the contemporaneous baseline;
- joint exact `>=0.60`;
- balanced or joint exact at least `+0.05` over matched recency-K16; and
- wrong-key valid-value swap fraction at least `0.10` below historical
  `0.8069306931`.

Surprise fit elapsed, post-warm wall, independently warmed training step and
peak allocation must each be no more than `1.25x` the contemporaneous native
FutureSeed baseline. Recency is also timed to expose admission-only overhead.

Any activation, quality, integrity or cost miss closes the entire sparse event
replay family. There is no K, temperature, admission score, cache width,
position encoding, receiver projection, seed, LR, loss, batch, width, depth,
epoch or duration rescue. A full pass authorizes one fused D256/L12 Sudoku
transfer; a miss returns research to a genuinely new recurrent transition.

## 8. Commands And Artifacts

The sole launcher is `scripts/run_zoology_gdn2_surprise_replay.sh`, configured
by `configs/retrieval/zoology_gdn2_surprise_replay.env`. Formal work requires
runtime `EXPECTED_GPU_NAME`, `EXPECTED_GPU_UUID` and exact pushed
`EXPECTED_SOURCE_SHA`. The launcher archives contract, logs, configs,
checkpoints, cases, scores, source snapshot, GPU evidence and SHA256 manifest
under `/huyang2/double-loop/runs/<run_name>`.

## 9. Decision

Pending the strict CUDA contract and the single fixed three-arm endpoint.
