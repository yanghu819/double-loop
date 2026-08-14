# P-GDN3-041: Query-Aware Live Delta GDN2

## 1. Metainfo

- Status: approved and preregistered; implementation pending
- First decision field: directional MQAR L1024 wrong-key binding regime
- Fixed model: D128/L2/H4/K32/V32, native FutureSeed
- Fixed data: four associations, 10,000 train and 1,000 validation examples
- Fixed optimization: 10 epochs, batch32, seed123, AdamW `1e-3`, weight
  decay `0.1`, cosine schedule, query-only cross entropy
- Arms: contemporaneous native GDN2 control followed by the sole candidate in
  one process on one GPU
- Candidate increment: one zero-initialized D128-to-H4 projection with bias per
  layer, exactly 516 parameters/layer and 1,032 total
- Candidate state/scan increment: zero state values, zero logical scans

## 2. Evidence And Hypothesis

The validated L1024 failure mode is mostly correct-value/wrong-key binding,
not absent values. P020's private Log-SPD address metric remains the only
large positive GDN3 signal, while V-only expansion, K expansion, fixed product
features, extra state slots, Raven address context and committed-residual
banks fail to close retrieval. P031 and P036 further show that moving erase
and update ownership to a second learned key destroys the coherent map.

Those failures do not test query-aware state evolution. Native GDN2 estimates
the content owned by key `k` and corrects that estimate, while the current
query `q` is used only after the update to read the state. The falsifiable
hypothesis is that `q` exposes a complementary binding error and must
participate in the same live token transition. If this is causal, a bounded
query term should improve both directional retrieval and joint exact while
reducing absolute errors and wrong-key swaps. If the path activates but those
metrics do not improve, query-feedback in this form is closed.

The mathematical inspiration is Q-Delta, but this is a clean-room vector-gated
GDN2 adaptation derived from the published recurrence. No source is copied
from the public Q-Delta repository, which has no repository license at the
audited commit `4afe5b5146c02acab0e59eb44929e77cfe9c6cf9`. Primary equation
source: arXiv `2606.08804`.

## 3. Fixed Recurrence

For each token and head, let native GDN2 produce unit-normalized `q,k`, K-wise
log decay `g`, K-wise erase gate `b`, V-wise write gate `w`, and value `v`.
With `D=Diag(exp(g))`, define

```text
lambda = 0.5 * tanh(W_lambda h + b_lambda)
x      = k + lambda * q
S_t    = D S_(t-1) - k [(b * x)^T D S_(t-1)] + k (w * v)^T
o_t    = q^T S_t / sqrt(K)
```

`W_lambda` and `b_lambda` are exactly zero at initialization. Therefore
`lambda=0`, `x=k`, and the recurrence is exactly native GDN2 while the
feedback parameters receive a first-order gradient. The fixed `0.5` bound
gives `lambda in [-0.5,0.5]`; because `q,k` are unit vectors,
`k^T x = 1 + lambda k^T q` lies in `[0.5,1.5]`. Signed feedback is required
for exact identity with a nonzero derivative at zero; a smooth nonnegative
coefficient cannot have both properties at an interior zero initialization.

The KxV form maps to the pinned official DPLR chunk operator with
`a=D*(b*x)`, `beta=-k`, additive address `k`, additive payload `w*v`, and
diagonal gate `g`. There is one official chunk scan per layer. The native
write/read/ownership address remains exactly `k`; no new state, cache, scan,
selector, rule, search or repair is introduced.

## 4. Boundary Versus Prior Failures

- P031/P036 changed an independent erase key and its rank-one update direction.
  P041 keeps the update and write direction exactly `k`; only the state content
  estimate adds the actual current query.
- P037 replaced vector erase/write control with a scalar committed residual.
  P041 retains every native K-wise erase and V-wise write gate.
- P007 read a completed incoming layer state before the scan and projected
  residuals into Q/K/gates. P041 changes every token transition using the live
  current-layer state.
- P039 used a second Raven state to rewrite Q/K. P041 has no second state and
  does not rewrite Q/K projections.
- P040 imposed a fixed product feature basis. P041 retains the learned native
  K32 geometry and adds no address dimensions.

This is a foundational recurrence, so it is trained from scratch on the
directional binding regime. A zero-initialized 100-step Sudoku graft is not a
valid first verdict.

## 5. Fixed Matched Protocol

The launcher accepts only CUDA index0 whose exact name and UUID match the
task. It requires a clean detached worktree at an exact pushed and read-back
GitHub SHA, pinned FLA SHA `9c8e42e...d85e`, clean Zoology SHA `1ad20d1`,
disabled backend dispatch, Triton short convolution, and no existing compute
process.

The strict CUDA checker must prove exact data hashes, parent parameter mapping,
zero feedback tensors, mathematical zero-feedback identity for zero and
nonzero incoming states, agreement with an explicit FP32 recurrence, pinned
official DPLR provenance, one DPLR backward path/layer, finite nonzero feedback
gradients, head permutation equivariance, query dependency, exact parameter
and state deltas, and no fallback.

The endpoint writes one matched parent initialization, then trains the native
control and candidate sequentially with identical data/order, optimizer,
schedule, batches, seed and evaluation cases. It saves cases, scores,
checkpoints, curves, warmed-step timing and peak allocation. There is no third
arm and no repeated control or seed.

## 6. Activation And Stability Gates

Every condition is required:

1. both layers and all eight layer/head feedback paths are active;
2. each layer has `lambda RMS >=1e-3`, finite nonzero board and token
   variation, and finite query-feedback relative RMS;
3. `max(abs(lambda)) <=0.5001` and every sampled `k^T x` lies in
   `[0.499,1.501]`;
4. the actual content-estimate change is finite and nonzero in both layers;
5. sampled transition spectral norm is finite and at most `1.35`;
6. terminal state RMS is finite in `[1e-4,1e4]`, varies by board, and is no
   less variable than `1e-6` across the fixed diagnostic boards;
7. native FutureSeed has exactly one active cross-layer route; and
8. the candidate adds exactly 1,032 parameters, zero state values and zero
   logical scans.

## 7. Quality And Cost Gates

All quality checks must pass simultaneously:

- balanced accuracy `>=0.85`;
- balanced gain over historical `.7475` `>=0.10` and over the contemporaneous
  control `>=0.10`;
- future and past accuracy each `>=0.85` and each gain over control `>=0.10`;
- joint exact `>=0.60` and gain over control `>=0.10`;
- total errors lower than control; and
- wrong-key valid-value swap fraction among errors lower by at least `0.10`.

All cost ratios to the contemporaneous control must pass:

- fit elapsed `<1.75x`;
- post-warm wall through checkpoint `<1.75x`;
- independent warmed-step elapsed `<1.75x`; and
- peak training CUDA allocation `<1.50x`.

Any integrity, activation, stability, quality or cost miss closes P041. There
is no lambda sign/cap, projection, normalization, gate, kernel, seed, LR, loss,
batch, width/depth, epoch/duration, FutureSeed or Sudoku rescue.

## 8. Commands And Artifacts

Planned launcher: `scripts/run_zoology_query_delta.sh`.
Planned config: `configs/retrieval/zoology_query_delta.env`.
Planned strict checker: `scripts/check_zoology_query_delta.py`.

The formal run will archive exact source and GitHub readback, GPU identity,
config, source snapshot, checker JSON/log, endpoint output, scores, cases,
checkpoints, timing, memory, exit classification and a SHA256 manifest under
`/huyang2/double-loop/runs/<run_name>`.

## 9. Decision

Pending the sole strict contract and fixed matched endpoint. A pass authorizes
one Sudoku transfer; a miss closes this query-feedback recurrence without a
nearby rescue.
