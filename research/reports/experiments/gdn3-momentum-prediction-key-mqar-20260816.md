# P-GDN3-063: Momentum Prediction-Key Decoupling

## 1. Research Question

Can P059 retain its stable Momentum write ownership while learning a separate
address for predicting the value already present in recurrent state, thereby
removing the residual adjacent-owner binding tail?

## 2. Evidence And Hypothesis

P-GDN3-059 is the strongest generic recurrent carrier in this workspace. On
directional MQAR L1024 it raises balanced accuracy from `.49400` to `.94425`,
joint exact from `.04100` to `.82400`, reduces errors from `2,024` to `223`,
and reduces wrong-key valid-value swaps from `1,546` to `151`. Native
FutureSeed carrying `[S,M]` is necessary, and the `M` component alone retains
essentially all of that cross-layer advantage.

The remaining errors are highly structured: all 151 swaps are adjacent in
write rank and 150/151 have the same direction. Directly replacing the learned
query with the exact own-write key collapses retrieval, and key-local old
Momentum is equally large in correct and swap cases. The model has therefore
learned a useful read coordinate and useful velocity; the unresolved issue is
the live prediction/commit binding.

The external Momentum recurrence already distinguishes a prediction key `p`
from the write/owner key `k`, but P059 leaves `p=None`, which makes the operator
use `p=k`. The hypothesis is that one address should estimate what state
already contains while another stable address should own the correction. A
learned `p` can subtract the interfering neighboring prediction without moving
the correction away from the native owner `k`.

## 3. Exact Mechanism

The unchanged recurrence is

```text
r_t = v_t - S_(t-1)^T (alpha_t p_t)
M_t = mu_t M_(t-1) - k_t (eta_t r_t)^T
S_t = alpha_t S_(t-1) - beta_t M_t
o_t = S_t^T q_t
```

P059 uses `p_t=k_t`. P063 adds one independent D128-to-D128 prediction-key
projection and one depthwise causal ShortConv4 per layer, then passes their
H4xK32 output only as `p_t`. Q, owner/write K, V, alpha, mu, beta, eta, output
projection, the sole Momentum scan, `[S,M]` geometry, and native FutureSeed
remain unchanged.

The new projection and convolution are loaded byte-identically from each
parent K path after the matched parent state is loaded. Thus `p_t=k_t` and the
complete model output and terminal state are exact at initialization, including
arbitrary finite nonzero incoming `[S,M]` state. The candidate adds exactly
`2 * (128*128 + 128*4) = 33,792` parameters and zero recurrent state values.
It uses the existing external fused chunk forward/backward, whose native API
already accepts distinct `p`; no recurrence source is copied or modified.

## 4. Distinction From Closed Families

- P031/P036 decoupled erase and write keys in first-order GDN2 and moved the
  committed owner direction. P063 keeps the sole owner/write `k` exact and
  decouples only residual prediction inside second-order Momentum.
- P060 transformed Q and K coherently. P063 leaves the trained read and owner
  geometries unchanged.
- P061 added a post-commit refresh microstep, while P062 replaced the
  recurrence with a custom lookahead backward. P063 adds no scan or custom
  recurrence and uses the production-stable external backward.
- It is not a cache, selector, side memory, router, parallel expert, reverse
  scan, Sudoku rule, readout repair or FutureSeed content transform.

## 5. Fixed Protocol

- directional MQAR L1024, four associations, D128/L2/H4/K32/V32;
- one from-scratch candidate, 10 epochs, batch32, seed123;
- exact P059 train/test data and exact shared matched initialization;
- frozen P059 score, cases and checkpoint are the only reference; no control
  rerun;
- native stacked `[S,M]` FutureSeed, one receiving route;
- one visible registered CUDA index 0 only;
- no projection/conv tie, scale, interpolation, rank, state, seed, LR, loss,
  batch, width, depth, duration or dataset sweep.

## 6. Strict CUDA Contract

Before formal training, exact pushed source in a clean detached worktree must
prove:

1. registered GPU UUID, exact external Momentum SHA and pinned host FLA SHA;
2. exact external layer/chunk/recurrent source hashes, the native fused
   `Chunkmode_ruleFunctionBackward`, and no fallback;
3. exactly `33,792` new parameters, zero recurrent-state and scan delta;
4. exactly two prediction projections and two depthwise ShortConv4 paths;
5. byte-exact P-to-K projection and convolution initialization after loading
   the exact matched parent;
6. full-model output and per-layer output/terminal-state identity to P059 with
   both zero and arbitrary finite nonzero incoming `[S,M]` state;
7. finite nonzero gradients through P projection/conv, Q/K/V/Momentum gates
   and the native FutureSeed edge on production-length L1024;
8. synthetically opening P changes output and terminal state, depends on P,
   preserves head permutation equivariance, and leaves owner K unchanged;
9. exact data/initialization hashes, finite bounded `[S,M]`, no NaN/OOM,
   concurrent model process or CPU model execution.

Any contract miss closes this implementation before formal training and opens
no kernel, projection, convolution, tie or tolerance rescue.

## 7. Falsifiable Prediction And Gates

Activation requires both prediction paths active; per-layer P/K relative RMS
divergence at least `.01`; finite nonzero board, token and head variation;
mean normalized P/K cosine in `[.80,.9999]`; all P projection/conv gradients
finite and nonzero; both Momentum layers and the native FutureSeed route
active; and per-layer Momentum-to-state RMS in `[.01,20]`.

Quality relative to frozen P059 passes only if every condition holds:

- balanced accuracy at least `.95500` and gain at least `.01000`;
- future and past accuracy each regress by at most `.00500`;
- joint exact at least `.84000` and gain at least `.01000`;
- total errors at most `178` and at least 20% below `223`;
- wrong-key valid-value swaps at most `120` and at least 20% below `151`;
- conditional wrong-key fraction falls by at least `.05`; and
- adjacent-write-rank swaps fall by at least 20%.

Cost relative to P059 requires elapsed, post-warm wall and independently warmed
step ratios each below `1.30x`, with peak allocation below `1.10x`.

Any activation, stability, quality or cost miss closes P063. There is no
projection/conv tie, interpolation, scale, rank, key-width, state, seed, LR,
loss, batch, model width/depth, duration or initialization rescue. Only a
complete pass authorizes one hard-Sudoku transfer.

## 8. Status

Preregistered before implementation. Source SHA, strict CUDA contract, formal
run and science verdict are pending.

## 9. Required Artifacts And Next Decision

Archive the contract, formal score/cases/checkpoint, exact frozen hashes,
validation curve, per-direction transition taxonomy, P/K divergence and state
diagnostics, independent throughput, memory and GPU telemetry, source/config/
log hashes, and GitHub readback.

A complete pass establishes a better GDN3 candidate and authorizes one hard
Sudoku transfer with native `[S,M]` FutureSeed. A miss closes prediction-key
decoupling and sends the next decision back to stable scalable live-state
organization rather than another address/readout patch.
