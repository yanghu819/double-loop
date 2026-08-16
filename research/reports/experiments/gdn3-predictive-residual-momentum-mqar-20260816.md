# P-GDN3-062: Predictive-Residual Momentum GDN3

## 1. Research Question

Can one internally consistent second-order commit remove P059's residual
adjacent-owner tail by evaluating the token residual at the state that old
Momentum is about to produce, instead of at the pre-Momentum state?

## 2. Evidence And Hypothesis

P-GDN3-059 is the strongest generic carrier so far. On directional MQAR L1024
it raises balanced accuracy from `.49400` to `.94425`, lowers errors from
`2,024` to `223`, and lowers wrong-key valid-value swaps from `1,546` to `151`.
Native FutureSeed is also causal: transporting only `M` retains balanced
accuracy within `.00050`, while removing `M` destroys future retrieval.

The frozen key-local diagnostic now shows why a simple erase is not justified.
At the causal write, old-velocity/current-update RMS is `12.53231` for swaps
and `12.38849` for matched correct cases, only `1.01161x` apart. Removing old
key-local Momentum reduces immediate residual strongly for both groups and is
slightly less favorable on swaps. Old Momentum is therefore a universal part
of the successful integrator, not an error-specific contaminant.

The remaining inconsistency is where the residual is measured. P059 predicts
the current value from the decayed base state, forms a gradient, and only then
applies the much larger old velocity. The current gradient cannot compensate
for a displacement it has not seen. The hypothesis is that the residual must
be evaluated at a one-step lookahead containing that already-committed old
velocity. This is the recurrent-memory analogue of a predictive or Nesterov
update; it keeps the useful velocity instead of erasing or retuning it.

## 3. Exact Mechanism

For normalized prediction key `p_t` and write key `k_t`, define

```text
A_t = exp(log_alpha_t)
U_t = exp(log_mu_t)
L_t = A_t S_(t-1) - beta_t U_t M_(t-1)
r_t = v_t - p_t^T L_t
M_t = U_t M_(t-1) - eta_t k_t r_t^T
S_t = A_t S_(t-1) - beta_t M_t
o_t = q_t^T S_t
```

P059 instead evaluates `r_t` against `A_t S_(t-1)` and only exposes old
velocity through the later `S_t` update. P062 changes only this evaluation
point. Q/K/P/V projections, alpha/mu/beta/eta constraints, short convolutions,
state shape `[S,M]`, native FutureSeed normalization/gate, model, data and
training schedule remain fixed.

The candidate adds zero parameters and zero recurrent values. It performs one
strictly causal scan and one committed update per token. A clean-room fused
Triton forward/backward implements the displayed recurrence from the equation;
no source from the unlicensed external Momentum checkout is copied into this
repository. The pinned external layer remains a read-only runtime dependency
for its projections and gate parameterization.

## 4. Distinction From Closed Families

- P061 appended a second residual-only microstep after the parent update,
  doubled transient tokens, and did not change `S` in that refresh. P062 has no
  refresh token, no second residual and no doubled scan. It replaces the
  primary one-step recurrence and is trained from scratch.
- P-DIAG-MOMVEL-001 removed old Momentum along the current key. P062 preserves
  all old Momentum and makes the new residual account for its impending state
  displacement.
- P060 and the decoupled-key/preconditioner families alter address geometry.
  P062 keeps the same learned Q/K/P coordinates and gates.
- It is not a cache, selector, replay, Raven router, parallel expert, side
  state, V-lifetime wrapper, Sudoku rule or post-scan readout.

The exact P061 microstep remains closed: no refresh count, order, coefficient
or tolerance is being retried.

## 5. Fixed Protocol

- directional MQAR L1024, four associations, D128/L2/H4/K32/V32;
- one from-scratch candidate, 10 epochs, batch32, seed123;
- exact P059 train/test data and exact shared matched initialization;
- frozen P059 score/cases/checkpoint are the only reference; no control rerun;
- native stacked `[S,M]` FutureSeed, one receiving route;
- one visible registered CUDA index 0 only;
- no coefficient, gate, state-size, seed, LR, loss, batch, width, depth,
  duration or dataset sweep.

## 6. Strict CUDA Contract

Before formal training, exact pushed source in a clean detached worktree must
prove all of the following:

1. registered GPU UUID, exact external Momentum SHA and pinned host FLA SHA;
2. exact external projection/gate source hashes and no generated fallback;
3. exactly zero parameter and persistent-state delta versus P059;
4. two clean-room fused forward and backward paths, one scan per layer;
5. FP32 Torch reference parity for outputs, terminal `[S,M]`, initial-state
   gradients and Q/K/P/V/alpha/mu/beta/eta gradients, including arbitrary
   finite nonzero incoming state;
6. exact parent-function identity when `beta=0` and when old Momentum is zero;
7. head and batch permutation equivariance and causal-prefix dependence;
8. finite nonzero gradients through every projection/gate and the native
   FutureSeed edge;
9. inverse-reconstruction denominator
   `abs(1-beta*eta*dot(p,k)) >= .05`, finite state, and `M/S` RMS in
   `[.01,20]`; and
10. exact data/initialization hashes, no NaN/OOM/concurrent model process, and
    no CPU model execution.

Any contract miss closes this implementation before formal training. It does
not open a precision, epsilon, kernel, coefficient or recurrence-order rescue.

## 7. Falsifiable Prediction And Gates

Activation requires both layers and the one FutureSeed route active, lookahead
displacement relative RMS at least `.01`, residual change relative RMS at least
`.01`, finite board/token variation, denominator at least `.05`, and bounded
noncollapsed `[S,M]` state.

Quality relative to frozen P059 passes only if every condition holds:

- balanced accuracy at least `.95500` and gain at least `.01000`;
- future and past accuracy each regress by at most `.00500`;
- joint exact at least `.84000` and gain at least `.01000`;
- total errors at most `180`;
- wrong-key valid-value swaps at most `105`;
- conditional wrong-key fraction at most `.60713`; and
- adjacent-write-rank swaps fall by at least 20%.

Cost relative to P059 requires elapsed, post-warm wall and independently warmed
step ratios each below `2.50x`, with peak allocation below `1.25x`. These limits
are fixed before implementation and include the clean-room recurrent backward.

Any activation, stability, quality or cost miss closes P062. There is no
lookahead coefficient, clipping, epsilon, gate, state, address, seed, LR, loss,
batch, width, depth, duration or initialization rescue. Only a complete pass
authorizes one hard-Sudoku transfer.

## 8. Status

Registered before implementation and before any candidate score was observed.

## 9. Required Artifacts And Next Decision

Archive the contract, formal score/cases/checkpoint, exact frozen hashes,
validation curve, per-direction transition taxonomy, lookahead/residual/state
diagnostics, independent throughput, memory and GPU telemetry, source/config/
log hashes, and GitHub readback.

A complete pass promotes Predictive-Residual Momentum as the GDN3 carrier and
opens one matched hard-Sudoku transfer with native FutureSeed. Any miss closes
this recurrence and returns to a genuinely different scalable state
organization; it cannot justify another Momentum microstep or lifetime sweep.
