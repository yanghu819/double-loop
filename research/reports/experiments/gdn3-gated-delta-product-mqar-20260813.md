# P-GDN3-029: Native Gated Delta Product With FutureSeed

## 1. Metainfo

- Status: preregistered; implementation and static checks in progress
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

Pending the one registered contract and one fixed training run.

## 7. Decision

Pending. Pass authorizes one hard-Sudoku transfer gate; failure closes native
n=2 GatedDeltaProduct plus FutureSeed and moves to a different state topology.
