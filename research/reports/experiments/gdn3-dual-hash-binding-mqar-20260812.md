# P-GDN3-024: Dual-Hash Binding Memory

## 1. Metainfo

- Status: approved, awaiting one fixed endpoint
- Benchmark: directional MQAR L1024 wrong-key regime
- Fixed setting: D128/L2/H4/K32/V32, native FutureSeed, 10 epochs,
  batch32, seed123
- Resource: one task-mode GPU, no concurrent model process

## 2. Evidence And Hypothesis

P020 shows that address geometry matters: a bounded Log-SPD metric raises the
current-runtime balanced score from `0.1735` to `0.48225`. Its remaining errors
are still `94.16%` wrong-key/valid-value swaps. P021 shows that a second write
to the same state overwrites useful content. P022/P023 show that ordinary H8
banks, with or without Log-SPD, cost more and do not close binding. P013 used a
separate zero-gated companion state and died before its companion addresses
received a production gradient.

The untested hypothesis is redundant binding in one live state: a value should
be retrieved only when two independently learned address subspaces agree. A
wrong value then needs to collide under both hashes, while the payload is still
written once by the same recurrent transition.

## 3. Mechanism

Keep the exact H4/K32/V32 state, projections, parameters and single pinned
official GDN2 scan. Reshape every projected Q and K head from K32 to two K16
factors and normalize them independently. Apply one fixed signed permutation
to the second factor, multiply the factors elementwise, concatenate that
compact bilinear sketch with a fixed cyclic shift, and normalize back to K32.
The resulting dot product is a zero-parameter degree-2 binding sketch: a wrong
value must collide in both factors to receive a strong score. V, decay, erase,
write, output, FutureSeed, optimizer and data are unchanged.

This is not H8 bank splitting: it does not create more physical heads, payload
streams, parameters, state bytes or kernel work. It is also not P020's learned
metric: the intervention changes the binding topology from one K32 similarity
to a compact product binding of two K16 factors.

## 4. Falsifiable Prediction And Gate

The one endpoint passes only if all hold:

- balanced accuracy is at least `0.60` and at least `+0.10` over P020
  `0.48225`;
- joint exact is at least `0.15`, and past/future are each at least `0.58`;
- wrong-key valid-value swap fraction among errors falls by at least `0.10`
  from P020 `0.941565`;
- both hashes have nonzero token variation and equal applied norm;
- parameter delta is zero, state remains 4,096 values/layer, and there is one
  official scan/layer;
- fit and allocation are at most `1.25x/1.10x` the current-A100 control.

Any miss discards dual-hash binding. There is no hash count/partition/order,
soft weighting, metric combination, head geometry, epoch, LR, loss, seed,
width, depth or duration rescue. A pass unlocks a fused Sudoku-scale test; a
failure closes fixed block-contiguous redundant addressing and redirects the
next decision to a genuinely learned sparse delta memory.

## 5. Configuration

- train/test examples: `10,000/1,000`
- sequence length / KV pairs: `1024/4`
- train tokens: `102.4M`
- source and official-FLA provenance: recorded by the launch artifact

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Decision

Pending the single registered endpoint.

## 9. Submission

Not applicable.
