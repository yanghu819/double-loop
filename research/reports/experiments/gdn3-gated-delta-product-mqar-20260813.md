# P-GDN3-029: Native Gated Delta Product With FutureSeed

## 1. Metainfo

- Status: completed; rejected at the registered quality and cost gate
- Decision field: directional MQAR L1024 binding closure before Sudoku transfer
- Fixed setting: D128/L2/H4/K32/V32, native FutureSeed, 10 epochs, batch32, seed123
- Resource: one task-mode A100-SXM4-80GB, CUDA index0

## 2. Evidence And Hypothesis

P-FS2-007 proves that retaining exact high-surprise evidence helps, but leaves
`99.8534%` of its remaining errors as correct-value/wrong-key swaps. P028 then
reduces swaps sharply by forcing a paired address, while destroying total
retrieval accuracy. These results say that both binding and a learnable
value/read map matter; another cache, address wrapper, or independent payload
is not justified.

The falsifiable hypothesis is that a native product of two learned delta
transformations inside one recurrent transition can separate binding while
preserving end-to-end value/read learning. Unlike a zero-initialized Sudoku
graft, this foundational recurrence must be trained from scratch on the
validated L1024 wrong-key regime.

## 3. Mechanism

Use the pinned official FLA `GatedDeltaProduct` layer with
`num_householder=2`, `use_forget_gate=True`, and
`allow_neg_eigval=False`. Each logical token applies one learned forget gate,
then two sequential learned key/value/beta delta transformations to the same
K32xV32 state, followed by one query read. The dedicated official
`chunk_gated_delta_product` Triton operator and its autograd are used without a
copied recurrence or fallback.

Both layers retain native terminal-state FutureSeed. The persistent state is
unchanged at 4,096 values per layer. The D128/L2 model has 666,200 parameters,
only 4,616 more than the 661,584-parameter native-GDN2 control.

## 4. Novel Boundary

P021 and P028 encoded extra microsteps around `chunk_gdn2`; P021's independent
payload overwrote the state and P028 analytically transformed addresses. P029
instead uses a distinct official operator whose two transformations are
jointly trained as one generalized delta-product transition. It does not add a
cache, state bank, selector, inverse Gram, parallel expert, reverse scan, or
task-specific rule. No prior local experiment has tested this FLA operator.

## 5. Registered Contract And Gates

Before training, the strict CUDA contract must prove the exact GPU UUID,
pinned FLA/Zoology and fixed-data hashes, exact official layer/operator source
hashes, two exact official layers, Triton short convolution, n=2, positive beta
range, 4,096 state values per layer, 666,200 total parameters, exactly two
`ChunkGatedDeltaProductFunctionBackward` paths, finite nonzero gradients for
both K/V/beta transform slices in both layers, nonzero incoming-state
dependency, transform-order noncommutativity, and no fallback.

The single fixed candidate passes only if all checks hold:

- both layers retain distinct learned transforms: key/value pair relative RMS
  `>=0.10`, beta-pair mean absolute difference `>=0.01`, key/value absolute
  pair cosine `<=0.98`, finite nonzero board-varying terminal states, and one
  native FutureSeed route;
- balanced accuracy `>=0.85`, future/past each `>=0.85`, joint exact `>=0.60`;
- balanced accuracy exceeds historical `0.7475` and the exact current-runtime
  control by at least `0.10` each;
- wrong-key swap fraction falls at least `0.10` below historical `0.806931`,
  and total errors are lower than both references;
- fit, post-warm wall, and warmed-step ratios are each `<=1.75x`; peak CUDA
  allocation is `<=1.50x` current-runtime control.

Any contract, activation, quality, or cost miss closes this mechanism without
Householder-count, signed-eigenvalue, beta, gate, scale, seed, LR, loss, batch,
epoch, width, depth, or Sudoku rescue.

## 6. Results

R1 stopped before model execution because the contract used `inspect.getfile`
on a `torch.compile`-wrapped export and therefore saw Torch Dynamo's wrapper
path. R2 changed only source-file discovery to verify the exported function
identity and the official module file; mechanism, data, protocol, and gates
were unchanged.

The R2 A100 contract passed. It verified the exact target UUID, both pinned
official `GatedDeltaProduct` layers, Triton short convolutions, exactly two
`ChunkGatedDeltaProductFunctionBackward` paths, 666,200 parameters, unchanged
4,096-value state, finite gradients through both K/V/beta transform slices,
one active FutureSeed gate, incoming-state dependency, and noncommutative
transform order. The two trained projection branches also passed every
registered divergence and bounded-state diagnostic. This establishes that the
two projection branches learned different tensors; it does not claim that
both branches made independently useful committed edits.

The fixed endpoint reached balanced/future/past/joint accuracy
`0.1370/0.1460/0.1280/0`, versus the same-runtime native-GDN2 FutureSeed
reference's `0.30625/0.3115/0.3010/0` and the locked historical reference's
`0.7475/0.7415/0.7535/0.339`. Candidate errors rose to `3,452`, from `2,775`
in the runtime reference. Wrong-key valid-value swaps fell from
`1,271 (0.458018 of errors)` to `584 (0.169177)`, but this is the same failure
mode exposed by P028: suppressing swap composition while replacing it with
more non-swap retrieval failures.

Fit/post-warm/allocation ratios were `1.3268x/1.3308x/0.8864x`; the warmed-step
ratio was `1.9634x`, above the registered `1.75x` ceiling. Cost is supporting,
not decisive, because its denominator is a frozen earlier A100 run and the
candidate retained inactive Python diagnostic hooks. The quality miss is
large enough to close the family independently of timing.

Formal run:
`p-gdn3-029-gated-delta-product-l1024-r2-20260813T020347Z-3c255e0`.
Source SHA is `3c255e0a4f206e8812d70bdf3d63d284329164f0`; decision, contract,
checkpoint, formal-log, and source-snapshot SHA256 are respectively
`ddbcdcee0041349c60ad645bde656da7a4b23c027ef79738074c499d3d9cd1a3`,
`d28f18dfdcdbd7c99ceaa84c25404c84addf6f4b4c474c1ebd4003309054d544`,
`5c00c64f2b93398ec1b7f4700dba450f60e4c8e493ac11dd3d187bcc981af0c3`,
`30d2caf8eba1b0b64a74e3e0cce51d6add573428d4cfa0bec667c2cd8d078d48`,
and `b8b0844d26a984d1042ee63bfd45636c6f5a3d8104cb3c00c67bfc0bd021a35c`.

## 7. Decision

Reject P-GDN3-029. Do not transfer it to Sudoku or rescue Householder count,
signed eigenvalues, beta, gate, projection, seed, LR, loss, batch, duration,
width, or depth. Native sequential transformations can reduce wrong-key swap
composition, but this endpoint does not preserve learnable retrieval. Move to
a different stable state topology that separates erase and write organization
without duplicating payload or wrapping the same KxV update.
