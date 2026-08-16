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

Complete and discarded. Exact GitHub source
`b41a0e74367d1d8df59304215612226fcc411e7f` was read back, checked out as a
clean detached worktree, and run on the sole A800 CUDA index 0
`GPU-c1d7c624-a393-befa-3807-7e00602d65ca`.

R1 completed every strict contract assertion but its outer launcher lost the
closed caller stdout after the Python contract exited successfully, so `tee`
returned 1 and the combined status became 74. No formal model started. This is
archived as a non-science orchestration failure. R2 redirected the unchanged
launcher output to a persistent file and reran the complete chain. The R2
contract passed:

- candidate/control parameters are `599676/599672`, exactly four new angles,
  with zero persistent-state and scan delta;
- zero-angle full output and arbitrary nonzero incoming/terminal `[S,M]` state
  are bit-exact to P059;
- both native Momentum layers, the native FutureSeed route and all four angle
  gradients are finite and nonzero;
- opening the phase changes the receiving output/state while owner K remains
  exact and head permutation error is zero; and
- FP32/BF16 energy relative error is `5.96e-8/.0004174`.

The formal endpoint fails quality decisively despite passing every integrity,
activation, stability and cost check:

| metric | frozen P059 | phase transport | delta |
|---|---:|---:|---:|
| balanced accuracy | .94425 | .25950 | -.68475 |
| future accuracy | .95150 | .24750 | -.70400 |
| past accuracy | .93700 | .27150 | -.66550 |
| joint exact | .82400 | .00100 | -.82300 |
| total errors | 223 | 2962 | +2739 |
| wrong-key valid-value swaps | 151 | 1081 | +930 |
| adjacent-write-rank swaps | 151 | 1056 | +905 |

All four angles activate: values are
`[-.097260,-.081167,-.008744,-.013114]`, mean/min absolute sine are
`.050011/.008744`, and the seed residual relative RMS is `.064694`. The
rotation remains numerically orthogonal (`2.37e-7` energy error), and both
state and Momentum RMS remain finite. However, phase residual board variation
is only `2.79e-9` while head variation is `.020295`: this is a global
head-specific channel rotation, not owner-specific evidence. Of 3,777 frozen
correct query events, only 981 stay correct and 2,796 break. The conditional
wrong-key share happens to fall because broad retrieval failure dominates; it
is not a binding improvement.

Elapsed/post-warm/warmed-step/peak-allocation ratios are
`1.01958/1.01823/.91544/1.00105x`. The candidate warmed benchmark is
`1.066M examples/s` and `1.092M tokens/s`. Across the complete R2 chain, 47
active five-second GPU samples average `58.36%` utilization and `157.66 W`,
with `82%`, `3,844 MiB` and `247.09 W` sampled peaks. Exit status 2 is the
registered science rejection, not an infrastructure failure.

## 9. Required Artifacts And Next Decision

The complete R2 run is
`/huyang2/double-loop/runs/p-fs2-014-momentum-phase-r2-20260816T041740Z-b41a0e7`.
SHA256 values are:

- contract JSON: `22622f849fcc4a27cd7d8b3c3feadae03c6170b9b10019e2b7a9ea9d6f098ff0`;
- comparison/score: `72d07eadf15c8b3fea698959db4ffefe0a547d1b55a4d73a15107d097ca64c46`;
- checkpoint: `46cce2194d06d682c32c469c9d0cc37019461ee66067614fd8ec59887c1cf495`;
- formal log: `ad245841eef8953f03295239039332a15c865cffddf0c3cb01dccf367435260f`;
- GPU telemetry: `f305deed95f2acb72b1d65d7b1c82e44c384ba5d90c6a335fe99e18de9fa6f77`;
- source snapshot: `b90455945818b74b19b363441696b990e9fda56d2e9843be3e2cc4598aa4bd04`.

Decision: close the entire S/M component-phase family. Do not retry angle
initialization, cap, sign, sharing, scale, seed, LR, loss, batch, width, depth,
duration or dataset. No Sudoku transfer is authorized. The result sharpens
P-DIAG-MOMFS: Momentum is useful future evidence, but `S` and `M` are not
exchangeable coordinates. A successor must preserve their dynamical semantics
and change a distinct live recurrent state organization or ownership
transition.
