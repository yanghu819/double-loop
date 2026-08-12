# P-GDN3-024: Dual-Hash Binding Memory

## 1. Metainfo

- Status: discarded after the single registered endpoint
- Source: exact pushed SHA `42a3557826c3d23c5d51731c4ed32945db8baca7`
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

- Run: `/huyang2/double-loop/runs/p-gdn3-024-dual-hash-l1024-20260812T080817Z-42a3557`
- Launch log: `/huyang2/double-loop/artifacts/launch/p-gdn3-024/formal-20260812T080817Z-42a3557.log`
- GPU: CUDA index0, `NVIDIA A100-SXM4-80GB`, UUID
  `GPU-0da20a4f-5e67-e47d-7aab-8c6efa2864ad`
- Pinned FLA source SHA: `9c8e42e762fce087c27b673af4922795d9edb85e`
- Checkpoint SHA256: `b31de349436936a72110d0ba611d8a529efc45dc1d9af11300e4c05bcdc911e2`
- Score/comparison SHA256: `e835b5ffe24919acb4ab1ac697b45fd331cfa3ab2a2820dc84dc43691142f724`
- Cases SHA256: `f1e5efc5402d5f6294dd76f23ed85e276909a52b7fa9fba2ee254443529381c6`
- Contract/abort SHA256:
  `48990216fb11b82861d7e89830b60f84b58a2ca6a8bf89cf4e862ec0426c0cc7` /
  `6f660babb15ea8b81a32ee6d9c4effa0616a857ad7ac1ee97cff2015e42f221c`

## 7. Results

The strict contract passes exactly as registered: both layers use the pinned
official `ChunkGDN2FunctionBackward`, parameter delta is zero, state remains
4,096 values/layer, and there is one official scan/layer. Both hash factors
are active. Endpoint token standard deviations span `0.06789..0.11963`, while
the maximum applied hash-norm imbalance is only `1.19e-7`.

| Metric | Result | Gate |
|---|---:|---:|
| balanced accuracy | `0.0115` | `>=0.60` and `>=0.58225` |
| future / past accuracy | `0.0105 / 0.0125` | each `>=0.58` |
| future / past CE | `4.91135 / 4.91902` | diagnostic |
| joint exact | `0` | `>=0.15` |
| wrong-key swaps / all errors | `120 / 3,954` (`0.03035`) | `<=0.84157` |
| fit ratio versus control | `0.81724x` | `<=1.25x` |
| allocation ratio versus control | `1.27185x` | `<=1.10x` |

The candidate does reduce P020's wrong-key valid-value fraction from
`0.94157` to `0.03035`. That apparent binding win is not useful retrieval:
accuracy collapses from P020 `0.48225` to `0.0115`, almost chance, while joint
exact falls from `0.044` to zero. The validation curve stays near chance in
all ten epochs and ends with CE `4.91581`, so this is not a late overfit or an
inactive-hash artifact. No NaN, OOM, fallback, source drift, data drift, GPU
drift, or concurrent model process occurred.

## 8. Decision

Discard P-GDN3-024. Multiplicative compact binding suppresses wrong-key
collisions by suppressing the usable learnable linear address channel itself.
The result closes fixed factor count, partition, signed permutation, cyclic
shift, soft interpolation, Log-SPD combination, and training-setting rescues.
The next credible mechanism must preserve the parent linear state as a usable
base path and add a bounded, independently testable correction memory rather
than replacing every native address with a product sketch.

## 9. Submission

Not applicable.
