# P-FS2-014: Momentum S/M Phase Transport

## 1. Research Question

Can native FutureSeed improve a strong second-order GDN recurrence by learning
how much transported Momentum should enter the receiver's state channel,
without changing address geometry, recurrent memory, scan math, or task logic?

## 2. Mechanism Hypothesis

P-GDN3-059 establishes that a primary Momentum recurrence plus native
FutureSeed solves the directional MQAR L1024 binding regime far better than
first-order GDN2. P-DIAG-MOMFS then isolates the cross-layer carrier: replaying
only normalized `M` gives `.94375` balanced accuracy versus `.94425` for the
native stacked `[S,M]` seed, while `S` only falls to `.48450`. The future
derivative is therefore useful, but the native receiver can only gate the two
components together and keeps them in their producer phase.

The hypothesis is that receiver computation benefits when a small learned
fraction of normalized terminal Momentum is rotated into its incoming state
component, while the complementary state evidence is retained in Momentum.
This is a receiver-native transport operation, not a new prediction key or a
change to the live token recurrence.

## 3. Exact Mechanism

Keep P059's external Momentum recurrence, all projections and gates, owner/
write key `K`, stacked terminal state `[S,M]`, and native shared FutureSeed gate
unchanged. Native FutureSeed first normalizes terminal `S` and `M` separately
to unit RMS over K/V and applies the existing receiver gate. For each receiving
head, add one angle `phi` and return

```text
S' = cos(phi) * S - sin(phi) * M
M' = sin(phi) * S + cos(phi) * M
```

Angles initialize to exact zero. Thus output and every incoming/terminal state
must be bit-exact to P059 at initialization, including arbitrary finite
nonzero incoming `[S,M]`. In D128/L2/H4, only layer 2 receives FutureSeed, so
the candidate adds exactly four scalar parameters. It adds no recurrent state,
scan, key, cache, convolution, projection, kernel, selector, or task rule.

Because the two components are normalized before rotation, this is a coherent
orthogonal transform in the S/M component plane. Per-head Frobenius energy is
preserved before BF16 rounding, and K/V owner coordinates are untouched.

## 4. Distinction From Closed Families

- P-FS3-004 rotates K/V state coordinates between layers. P-FS2-014 leaves the
  full K/V basis exact and rotates only the dynamical components `S` and `M`.
- P063 introduces an unconstrained prediction address and destroys P059's
  stable owner coordinate. P-FS2-014 adds no address at all.
- P-DIAG-MOMFS masks components and changes no learned transport. It motivates
  this run but does not test receiver-learned phase.
- This is not a producer codec, cache, router, side memory, reverse scan,
  recurrence coefficient, Sudoku heuristic, or inference repair.

## 5. Fixed Protocol

- directional MQAR L1024 with four associations;
- D128/L2/H4/K32/V32, 10 epochs, batch32, seed123;
- exact P059 train/test data and shared matched initialization;
- one candidate trained from scratch; frozen P059 score, cases and checkpoint
  are the only reference and the control is not rerun;
- exact external Momentum and host FLA provenance used by P059;
- one visible registered CUDA index 0 only;
- no angle initialization, cap, sign, sharing, tie, scale, seed, LR, loss,
  batch, model width/depth, duration, data, or threshold sweep.

## 6. Strict CUDA Contract

Before formal training, exact pushed source in a clean detached worktree must
prove:

1. registered GPU UUID, external Momentum SHA, host FLA SHA, source hashes and
   native fused `Chunkmode_ruleFunctionBackward`, with no fallback;
2. exact P059 data and initialization hashes;
3. exactly four new parameters and zero recurrent-state/scan delta;
4. full-model logits, every layer output, and every terminal `[S,M]` state are
   bit-exact to P059 at `phi=0`, for zero and arbitrary finite incoming state;
5. all four angles and P059 Q/K/V/gates plus the native FutureSeed edge receive
   finite nonzero gradients on production-length L1024;
6. opening a synthetic angle changes the receiving output and state, preserves
   head permutation equivariance, and leaves owner K byte-exact;
7. FP32 per-head rotation energy relative error at most `1e-5` and BF16 at
   most `5e-3`, with finite bounded state and Momentum;
8. no NaN, OOM, concurrent model process, CPU model execution, or silent
   fallback.

Any miss closes this implementation before formal training and opens no
contract-tolerance or implementation rescue.

## 7. Falsifiable Prediction And Gates

Activation requires all four angles finite and nonzero, mean absolute
`sin(phi)` at least `.01`, minimum per-head absolute `sin(phi)` at least
`.001`, finite nonzero board/head variation, seed residual relative RMS at
least `.01`, and finite per-layer state/Momentum RMS with `M/S` in `[.01,20]`.

Quality relative to frozen P059 passes only if every condition holds:

- balanced accuracy at least `.95500` and gain at least `.01000`;
- future accuracy gain at least `.01000`;
- past accuracy regression at most `.00300`;
- joint exact at least `.84000` and gain at least `.01000`;
- total errors at most `178` and at least 20% below `223`;
- wrong-key valid-value swaps at most `120` and at least 20% below `151`;
- conditional wrong-key fraction falls by at least `.05`; and
- adjacent-write-rank swaps fall by at least 20%.

Cost relative to P059 requires elapsed, post-warm wall, and independently
warmed step ratios each below `1.10x`, with peak allocation below `1.05x`.

Any activation, stability, quality, or cost miss closes component-phase
transport. There is no angle initialization/cap/sign/sharing/tie/scale, seed,
LR, loss, batch, width, depth, duration, or dataset rescue. Only a complete
pass authorizes one fixed hard-Sudoku transfer.

## 8. Status

Preregistered. Implementation, strict CUDA contract, and the single fixed
candidate remain pending. No science score exists yet.

## 9. Required Artifacts And Next Decision

Archive the exact source/config/data/initialization hashes, contract JSON,
formal comparison, checkpoint, cases, log, GPU telemetry, per-head angles,
seed residual and energy diagnostics. Update `plans.md`, this report,
`leaderboard.csv`, `docs/PAPER_PLAN.md`, and `docs/LESSONS.md`, then commit and
push the result.

A pass selects receiver-native S/M phase transport as FS2 and opens exactly
one hard-Sudoku transfer. A miss closes this entire component-phase family and
returns the next decision to a distinct live recurrent state organization;
it does not authorize a nearby parameter or training sweep.
