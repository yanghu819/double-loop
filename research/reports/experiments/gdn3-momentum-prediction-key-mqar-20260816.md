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

Complete and discarded.

R1 source `7e32793db986de0cde1ce2d3f0cbcea420a590ff` stopped in the
contract before the first GPU model forward. The initial implementation used a
deep-copied `nn.Linear` for P. Zoology's global initializer therefore
reinitialized that copy and consumed RNG before construction of the next
native layer, changing parent tensors outside the candidate path. The parent
tensor invariant caught this. This is a non-science harness failure, not a
mechanism result; its contract-log and `abort.json` SHA256 values are
`79cf3b15f4ed4869a817f10fe822c4a7a0d7914cdae80ea8e17ab3dddb283038`
and `9c005b7be036e9f0b42293ec14a0f19173bda70a0b96976f8546216dc55d5fc0`.

R2 changes no mechanism or registered gate. Exact pushed/read-back source
`50009bd24fda09b2b50b9499c9b2fa6f027c2c44` represents P with a one-weight
module using `F.linear`, excludes it from the generic Linear initializer, and
then byte-copies K as originally specified. The clean detached A800 contract
passes:

- full-model logits and every per-layer output and terminal `[S,M]` state are
  bit-exact to P059 for zero and finite nonzero incoming state;
- parameter count is exactly `599,672 -> 633,464`, with four new tensors and
  no state or scan delta;
- both layers use native `Chunkmode_ruleFunctionBackward` and exact external
  Momentum SHA `c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`;
- P projection/conv, Q/K/V, alpha/momentum/erase/write, both recurrent layers
  and the receiving FutureSeed edge all have finite nonzero gradients;
- opening P changes output/state and preserves exact head equivariance; and
- train/test hashes are
  `647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68` /
  `4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f`.

The formal endpoint
`p-gdn3-063-momentum-prediction-key-r2-20260816T032241Z-50009bd`
completed normally with science exit status `2`. Frozen P059 versus candidate
is:

| Metric | P059 | P063 |
|---|---:|---:|
| balanced accuracy | .94425 | .00900 |
| future accuracy | .95150 | .01100 |
| past accuracy | .93700 | .00700 |
| joint exact | .82400 | 0 |
| total errors | 223 | 3,964 |
| wrong-key valid-value swaps | 151 | 131 |
| adjacent-write-rank swaps | 151 | 61 |

The lower swap count is not a repair. Paired transitions contain `3,620`
parent-correct to unrelated-wrong and `124` parent-correct to wrong-key
changes, while only three parent swaps become correct. Validation accuracy is
approximately chance for all ten epochs (`.00975` at epoch0 and `.00900` at
epoch9).

The mechanism activates destructively. Layer-0/1 P/K relative RMS divergence
is `1.69070/1.13315`, while mean normalized cosine is only `.15248/.14325`, far
below the registered `.80` floor. Layer 0 state and momentum RMS overflow to
`Inf`; its momentum/state ratio and the transported raw FutureSeed RMS are not
finite. Layer 1 remains finite but cannot recover the destroyed first-layer
trajectory. Activation/stability, every substantive quality route, and the
time-cost gate fail. Elapsed/post-warm/warmed-step ratios are
`1.59575/1.59904/1.55312x`; peak allocation is `1.03407x`.

The run sampled 89 active-memory telemetry points: mean/p95/peak utilization
`32.61/79/92%`, peak observed memory `3,970 MiB`, mean power `124.47 W`, and
peak power `236.70 W`. It ended with zero GPU memory and no compute process.
No NaN/OOM/fallback appears in the training log; non-finite recurrent
diagnostics are themselves the registered stability failure.

## 9. Required Artifacts And Next Decision

The remote run directory is
`/huyang2/double-loop/runs/p-gdn3-063-momentum-prediction-key-r2-20260816T032241Z-50009bd`.
Key SHA256 values are:

- contract JSON: `93f89af09ec0f414feabf1c7e3c5535e50bfa5e45aac98321fd9f481b4ee1cc2`;
- comparison/score JSON: `30d7630211794495c2185792d5af1a2eed9283f7514c51a2464e7405e3b90018`;
- candidate checkpoint: `9ddb505040852c212fa771810fe69034e065f0ec2cd78e8cb2d6b07a4f3f6605`;
- cases: `05b8382c719bb93c9ffb700e15eb4391082c199d052d7929c3906d35585ad60e`;
- formal log: `e507aced8be0c05ed36895af76d10fe6e1d4d432fc13f8c621c5c85678054ebc`;
- GPU telemetry: `6e3c5cd46d3db5778478d12cf7494ae06756091e02e56ef725d978fbeb243e41`;
- source snapshot: `a14030793c9078f7d7ece231b087b737de1a2b7be0fc064e445788b5f50aa7fd`.

Independent prediction-key decoupling is closed. P063 shows that the shared
prediction/owner address is a stabilizing coordinate constraint, not merely an
unnecessary parameter tie. Do not rescue it with tying, interpolation, scale,
rank or training sweeps. The next decision returns to stable scalable live
state organization or an FS mechanism that transports genuinely distinct
second-order evidence without changing the owner coordinate.
