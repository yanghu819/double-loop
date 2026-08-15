# P-GDN3-054: Block-RLS Constrained Delta GDN3

## 1. Metainfo

- Status: complete; discarded
- Date: 2026-08-15
- Branch: `codex/gdn3-block-rls-20260815`
- Exact source: `6b4f0c4ecabf2a82691c9535c1770c2f386a26a6`
- Formal run: `p-gdn3-054-block-rls-r2-20260815T165534Z-6b4f0c4`
- Decision: closed; no Sudoku transfer or same-family rescue
- First decision field: directional MQAR L1024 wrong-key binding regime
- Frozen control: deterministic P-REPRO-001 replay B
- Fixed model: D128/L2/H4/K32/V32 plus native FutureSeed
- Fixed data/training: four associations, 10,000 train and 1,000 validation
  examples, 10 epochs, batch32, seed123
- GPU: one task-mode CUDA index 0 only

## 2. Evidence And Hypothesis

The reproducible native endpoint reaches balanced/future/past/joint
`.494/.454/.534/.041`. Of its 2,024 errors, 1,546 (`76.38%`) are valid values
belonging to another key, and `99.61%` of those swaps choose the adjacent owner
by write rank. P-DIAG-EDIT-001 shows that deleting the wrong competing write
repairs `98.25%` of swaps but destroys the competing owner's own retrieval.
The primary problem is therefore destructive address interference inside one
live state, not absent values.

Directly separating erase and write keys is not the answer. P-GDN3-031,
P-GDN3-036, P-GDN3-043 and P-GDN3-053 all diversify address geometry but
collapse retrieval, while the exact dual-address oracle reaches only `.23025`.
Query, erase, write and read coordinates are jointly learned. Dense online
inverse geometry P-GDN3-027 had no quality verdict because its unfused KxK
token loop was at least `40.95x` control. The diagonal P-PCOND-001 produced a
small Sudoku blank gain but could not close exact boards and was too costly.

The falsifiable hypothesis is narrower: most harmful correlation can be
removed by an online block-sparse inverse-information state. Four-coordinate
blocks retain cross-coordinate information missing from a diagonal
conditioner, while reducing dense O(K^2) geometry to O(K*4). End-to-end Q/K
projections can learn which features should share a block. If this is the
right state organization, it should reduce adjacent-owner swaps without
breaking native retrieval and should run near the existing chunk path.

## 3. Mechanism

Split each normalized K32 address into eight fixed four-coordinate blocks.
For every board, head and block, initialize `P_0=I_4`. At token `t`, use the
native coordinate decay `A_t=Diag(exp(g_t))` and update:

```text
B_t = A_t P_(t-1) A_t^T + (I - A_t^2)
u_t = B_t k_t
d_t = 1 + k_t^T u_t
P_t = B_t - u_t u_t^T / d_t
```

The block scan is a dedicated forward-only Triton statistic. Its history is
stop-gradient, while the current key receives the exact local Jacobian
`B_t`; this is analogous to online normalization statistics and avoids an
unbounded reverse sequential graph. The eight block outputs are concatenated
to `u_t`.

Let the native erase address be `z_t=b_t*k_t`. Project `u_t` onto the exact
parent-response affine constraint:

```text
a_raw = u + z * (z^T k - z^T u) / ||z||^2
a     = k + tanh(alpha_head) * (a_raw - k)
```

with the finite `u` branch used when erase energy is negligible. Then make one
pinned-official DPLR chunk call:

```text
D   = Diag(exp(g)) S_(t-1)
e   = w*v - z^T D
S_t = D + a e^T
```

implemented by DPLR factors `alpha=A*z`, `beta=-a`, write key `a`, and payload
`w*v`. Because `z^T a_raw=z^T k`, the correction preserves the native erase
response while redistributing the same committed residual. Every mix scalar
starts at zero, so the full model is parent-equivalent at launch. For
D128/L2/H4 it adds exactly eight parameters and 512 transient FP32 geometry
values per layer, or 12.5% of the main recurrent state. Geometry resets at
each layer call and is not transported by FutureSeed.

## 4. Boundary

This is not the user-linked direct decoupled-key toy: there is one coherent
native ownership address and one constrained committed edit. It is not dense
P-GDN3-027 kernel rescue: the state organization is block sparse, has one
eighth the auxiliary state and asymptotically linear cost for fixed block
size. It is not P-PCOND-001's diagonal multiplier, P026's completed-block query
wrapper, a Raven router, cache, second payload bank, selector, Sudoku rule, or
post-scan readout. It changes each token's live committed edit and is trained
from scratch on the validated generic binding regime before any Sudoku use.

## 5. Strict CUDA Contract

Exact pushed source in a clean detached worktree must prove:

1. one visible CUDA index 0 with the registered UUID;
2. pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. one official DPLR chunk recurrence per layer with
   `ChunkDPLRDeltaRuleFunctionBackward`, Triton ShortConv and no fallback;
4. the custom block-statistic Triton kernel matches an FP32 Torch recurrence
   in transported blocks, preconditioned addresses, terminal precision and
   minimum denominator;
5. zero mix preserves full model output and both terminal states against the
   native GDN2 parent within the preregistered DPLR parity tolerance, including
   arbitrary finite nonzero incoming states;
6. exactly eight added parameters and 512 transient geometry values per layer;
7. all eight mix gradients and parent Q/K/V/g/b/w gradients are finite and
   nonzero; the current-key local Jacobian is present;
8. opened paths change output and state, depend on causal prefix keys, obey
   block/head permutation equivariance and satisfy the committed-response
   constraint;
9. every denominator is `>=1e-4`, terminal block precision is symmetric
   positive definite with eigenvalues in `[1e-4,1.001]`, and all values are
   finite; and
10. source/data/GPU provenance is exact with no CPU model path, concurrent
    model/eval, NaN, OOM or silent fallback.

## 6. Fixed Science Gate

After the contract, run one candidate-only 10-epoch/batch32/seed123 arm from
the SHA-locked P-REPRO-001 initialization. Do not repeat the frozen control.

Activation and stability require all eight paths active, mean absolute mix
`>=1e-3`, actual write-address relative RMS `>=.02`, terminal block off-diagonal
ratio `>=1e-3`, finite board variation, minimum denominator `>=1e-4`, maximum
constraint error `<=1e-4`, and all terminal precision eigenvalues inside the
contract bounds.

Quality requires all of:

- balanced accuracy `>=.65` and gain over control `>=.10`;
- future and past accuracy each `>=.62`;
- joint exact `>=.15` and gain over control `>=.10`;
- total errors at least `20%` lower; and
- wrong-key valid-value swap fraction among errors at least `.10` lower.

Cost requires fit elapsed, post-warm wall and independent warmed-step time
each `<1.75x` control, with peak allocation `<1.50x`. Any integrity,
activation, stability, quality or cost miss closes P-GDN3-054. There is no
block size, grouping, statistic-gradient, mix cap/init, prior, epsilon, seed,
LR, loss, batch, width/depth, epoch, duration, FutureSeed or Sudoku rescue.

## 7. Endpoint Result

The R2 strict contract passes. It verifies the pinned FLA source, two official
`ChunkDPLRDeltaRuleFunctionBackward` operators, one scan per layer, matching
Triton/Torch block recurrences, exactly eight new parameters, zero persistent
state, finite nonzero gradients, causal-prefix dependence, permutation
equivariance and stable precision geometry. The first R1 attempt stopped before
science because its checker treated an official `Sequential` projection as a
single `.weight`; the checker-only fix is the R2 source above and does not
change the mechanism, data, training or gates.

The fixed 10-epoch endpoint rejects the hypothesis:

| metric | frozen control | candidate |
| --- | ---: | ---: |
| balanced accuracy | `.494` | `.018` |
| future accuracy | `.454` | `.022` |
| past accuracy | `.534` | `.014` |
| joint exact | `.041` | `0` |
| total errors | `2024` | `3928` |
| wrong-key valid-value swaps | `1546` | `190` |
| swap fraction among errors | `.763834` | `.048371` |

The lower swap count is not a binding improvement. Paired predictions show
1,849 control-correct queries becoming another wrong value, while only 24
control wrong-key swaps become correct. Validation remains near chance through
epoch 5 and ends at `.018`, whereas the frozen control makes its learning
transition at epoch 5 and ends at `.494`.

The precision statistic is numerically healthy and all eight learned mix paths
are active. Mean absolute mix is `.02589/.02395` by layer, terminal block
off-diagonal ratio is `.08559/.04560`, minimum denominator is above `1.0`, and
terminal eigenvalues remain in `[.04038, 1.000001]`. However, actual address
relative RMS is only `.00772/.00796`, below the registered `.02` activation
floor, with very small board variation. Exact committed-response error remains
`2.38e-7`.

Elapsed, post-warm wall, independent warmed-step and peak-allocation ratios are
`1.7176/1.7266/2.0067/1.2732x`. The first three fit/allocation checks pass, but
the warmed-step `<1.75x` gate fails. Across 101 five-second GPU samples, the
all-sample mean is `44.26%`; the 67 active samples average `66.72%`, peak at
`95%`, and observe 2,339 MiB peak memory on the sole A100-SXM4-80GB.

## 8. Decision And Provenance

P-GDN3-054 is closed. Preserving the native erase scalar while changing the
committed write direction is not enough to preserve the jointly learned
query/erase/write/read closure. The result is consistent with direct,
anchored, biorthogonal and redundant-bank decoupled-key failures: address
decorrelation can reduce recognizable swaps by making the value code stop
working. Do not tune block size, grouping, statistic gradient, mix, prior,
epsilon, seed, optimization, model size, training duration or FutureSeed, and
do not transfer this mechanism to Sudoku.

- comparison SHA256: `f2259362b8d9c06f2972b90b5ea3907a80d08d1f51ee70b457473813bfd7ddb3`
- contract SHA256: `c9cbe3378d3ea3bb47a369bcc5a13a2ff5e3d722c88cbdb79e80cbd38808bacf`
- checkpoint SHA256: `4f36b6b9ea01a587dc848893307a92e389e83f2f31bb430877acd0e7cf9325a9`
- formal log SHA256: `40c4c5fa0d99655fbeae63f47779984a4bbc1cab592cd4b42d251115b6169e80`
- source snapshot SHA256: `880d80f4dda6ad43430c44bd522cbb14b9f19d7c9353435a7b535cbb3a7a74ac`
- science endpoint status: `0`; registered decision exit: `2` (gate fail)

The next architecture decision must preserve pair identity as a first-class
learned state organization, or carry receiver-native ownership evidence. It
must not be another post-hoc key transform, erase/write decoupler, fixed bank
average, preconditioner wrapper or extrapolation of an already merged state.
