# P-GDN3-027: Online Inverse-Gram Preconditioned Delta GDN

## 1. Metainfo

- Status: preregistered; no science result yet
- First decision field: directional MQAR L1024 wrong-key binding regime
- Fixed model: D128/L2/H4/K32/V32, native FutureSeed, one recurrent scan
- Fixed data: four associations, 10,000 train and 1,000 validation examples
- Fixed optimization: 10 epochs, batch32, seed123, AdamW `1e-3`, weight
  decay `0.1`, cosine schedule, query-only cross entropy
- Arms: contemporaneous parent control followed by OIG candidate on one GPU
- Candidate increment: eight zero-initialized per-layer/head mix parameters
  and one FP32 K32xK32 inverse-information state per head (`4,096` values per
  layer)

## 2. Evidence And Hypothesis

Directional MQAR L1024 isolates a binding error rather than missing values.
The historical validated carrier produced mostly correct-value/wrong-key
swaps. On the current runtime, P-GDN3-020 raised balanced accuracy from
`0.1735` to `0.48225`, yet joint exact remained `0.044` and `94.1565%` of
remaining errors were still wrong-key valid-value swaps. V-only capacity did
not fix the regime.

The falsifiable hypothesis is that a single normalized key direction is a poor
rank-one edit address after many correlated writes. Maintaining an online
inverse key Gram can whiten each new edit direction in the live recurrence,
while an exact linear constraint preserves the parent GDN2 committed residual.
If address interference is causal, the preconditioned address Gram should
decorrelate and wrong-key swaps should fall before any Sudoku transfer.

`P` below is an inverse-information/covariance state. It is not a precision
matrix, a learned metric parameter, or an extra payload bank.

## 3. Corrected Constrained Recurrence

All vectors are per token and head. The official path supplies normalized key
`k`, decay `g`, erase gate `b`, write gate `w`, and value `v`. Let

```text
A = Diag(exp(g))
D = A S_(t-1)
z = b * k
e = w * v - z^T D
```

where `*` is elementwise multiplication. `e` is the actual committed GDN2
residual; it is not a learned surprise score. Propagate the inverse-information
state with a forgetting prior whose fixed point is identity:

```text
B  = A P_(t-1) A^T + (I - A^2)
u  = B k
q  = 1 + k^T u
P_t = B - u u^T / q
```

Then project the preconditioned direction onto the exact parent-response
constraint and use a function-preserving learned interpolation:

```text
c = z^T k
a = u + z (c - z^T u) / ||z||^2,  if ||z||^2 > 1e-12
a = u,                              otherwise
m = tanh(alpha)
a_bar = k + m (a - k)
S_t = D + a_bar e^T
```

The correction fixes three earlier formulation errors. It does not call `P`
precision; it uses the covariance-form forgetting update; and it never uses
`z=b*k` as the sole write direction. Therefore `b=0` still writes through
`u`. When `P=I`, `B=I` and `a=k`, giving the parent first-token edit exactly.
For nonzero erase, `z^T a=z^T k`, so the response at the parent's erase
address and the committed residual `e` are preserved exactly for every `m`.
All eight `alpha` values initialize to zero, making the recurrence
mathematically parent-equivalent at step zero while allowing each layer/head
to learn how strongly it redistributes the update along the inverse-Gram
direction. The contract checks the actual compiled BF16 path against the
official kernel within a frozen numerical tolerance; it does not claim
bitwise equality between different reduction orders.

`B`, `u`, `q`, `P`, the constraint dots/outer products, and recurrent `S`
accumulation are FP32. `P` is symmetrized after the rank-one update. There is
no eigenvalue clipping or silent fallback: a Cholesky, eigenvalue, denominator,
or finite-value violation closes the run. The formal implementation uses a
fixed 16-token full-graph `torch.compile` Inductor CUDA chunk, called
sequentially over L1024. The eager Python token loop is its mathematical
source/reference only; formal training must prove a compiled full graph with
no eager fallback. A dedicated fused Triton kernel is deferred unless this
first trainable recurrence falsifier passes quality.

## 4. Boundary Versus P020-P026

- P020 learned one static bounded Log-SPD Q/K metric. OIG has no learned
  metric and changes each live state edit using all preceding keys. OIG adds
  only eight interpolation scalars; the inverse-information state is not a
  learned parameter.
- P021 issued a second destructive write to the same state. OIG keeps one
  committed residual and one constrained edit.
- P022/P023 added or partitioned payload state banks. OIG adds only a KxK
  inverse-information state and directly updates the one main KxV state.
- P024 compacted addresses into two fixed hashes. OIG retains the full K32
  address and adapts continuously from observed key geometry.
- P025 stored committed deltas in a second correction state. OIG preserves
  the committed delta and changes its address in the primary recurrence.
- P026 conditioned queries from a completed-block Gram without changing the
  token scan's live transition. OIG updates `P` and `S` at every token.

This is not PIC/HYPIC serving reuse, a surprise cache, Raven routing, a
FutureSeed producer codec, a Sudoku rule, search, repair, or selector. It is a
from-scratch foundational recurrence test on the validated binding regime.

## 5. Fixed Matched Protocol

The launcher accepts only one visible CUDA index 0 whose exact name and UUID
match runtime admissions. It requires the exact pushed source SHA in a clean
detached worktree, pinned FLA SHA `9c8e42e...d85e`, clean Zoology SHA
`1ad20d1`, disabled backend dispatch, Triton convolution, no existing compute
process, and SHA-locked P020 score/cases.

The strict CUDA checker runs first. It must prove Inductor full-graph code
generation with no eager fallback, parent parameter identity, exact
first-token identity from `P=I`, nonzero-incoming-state behavior,
compiled/reference forward and backward agreement,
finite gradients, head permutation equivariance, state dependency, committed
residual/constraint identity, the `b`-near-zero finite-write case, exact
eight-parameter delta, exact extra-state count, and one logical scan.

The endpoint then creates the parent control and OIG candidate sequentially.
Both use the same pre-generated train/validation hashes and order, the same
step-zero parent parameter tensors, optimizer, schedule, batches, seed and
evaluation cases. RNG is reset before each arm. Control is not historical
P020: it is the unchanged pinned-official GDN2 plus native FutureSeed on the
same runtime. P020 remains a SHA-locked external mechanism floor. Both arms
save cases, score, checkpoint, curves, warmed-step timing and peak allocation.

Endpoint exit `0` means every gate below passed. Exit `3` means a complete
registered science close; it is archived without `abort.json` and authorizes
no rescue. Any other nonzero status is an integrity/infrastructure error and
must write `abort.json`.

## 6. Activation And Stability Gates

Every condition is required:

1. all eight layer/head OIG paths are active;
2. each path has terminal
   `||offdiag(P)||_F / ||P||_F >= 1e-3`;
3. each layer has terminal validation aggregate
   `sqrt(sum||a-k||^2/sum||k||^2) >= 0.02`;
4. all state, address, residual and gradient diagnostics are finite, and every
   Sherman-Morrison denominator is at least `1e-4`;
5. maximum absolute symmetry error of `P` is at most `1e-5`;
6. every sampled eigenvalue of symmetrized `P` lies in
   `[1e-4, 1.001]`, with Cholesky success at every registered sample;
7. maximum constraint-identity absolute error is at most `1e-4`, and the
   near-zero erase contract retains a finite nonzero committed write; and
8. the extra state is exactly `H*K*K=4,096` FP32 values per layer, with
    exactly eight learned mix parameters total and one logical token scan.

## 7. Quality And Cost Gates

All quality checks must pass simultaneously:

- balanced accuracy `>=0.60`;
- balanced gain over the contemporaneous control `>=+0.20`;
- balanced gain over P020 `0.48225` `>=+0.10`;
- past and future accuracy each `>=0.58`;
- joint exact `>=0.15`; and
- wrong-key valid-value swap fraction among errors
  `<=0.8415741187831965`, exactly `0.10` below P020.

All cost checks are ratios to the contemporaneous control and must pass:

- trainer fit elapsed `<2.50x`;
- post-warm arm wall time through checkpoint `<2.50x`;
- independent warmed-step elapsed `<2.50x`; and
- peak training CUDA allocation `<1.80x`.

Any activation, stability, quality, cost or integrity miss closes OIG. There
is no inverse prior, forgetting, epsilon, projection, clipping, kernel, seed,
LR, loss, batch, width/depth, epoch, duration, FutureSeed, or Sudoku rescue.

## 8. Commands And Artifacts

The sole launcher is `scripts/run_zoology_gdn2_oig.sh`, configured by
`configs/retrieval/zoology_gdn2_oig.env`. Runtime must provide
`EXPECTED_GPU_NAME`, `EXPECTED_GPU_UUID`, and the exact pushed
`EXPECTED_SOURCE_SHA`.

Each run archives launch/config JSON, exact source HEAD/status/patch and source
snapshot, GitHub ref readback, GPU identity before/after, checker JSON/log and
pipeline status, endpoint log/status, arm outputs/checkpoints, final score,
terminal classification, and a SHA256 manifest under
`/huyang2/double-loop/runs/<run_name>`.

## 9. Decision

Pending one fixed matched endpoint. A pass authorizes analysis of OIG transfer
to hard Sudoku; a registered exit-3 close ends this exact mechanism before any
Sudoku run. No result may alter the gates above.
