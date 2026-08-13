# P-GDN3-030: Contractive Separate-Erase/Write DPLR

## 1. Metainfo

- Status: completed; rejected at the registered quality gate
- Decision field: directional MQAR L1024 wrong-key binding before Sudoku transfer
- Fixed setting: D128/L2/H4/K32/V32, native FutureSeed, 10 epochs, batch32, seed123
- Resource: one task-mode A100-SXM4-80GB, CUDA index0

## 2. Evidence And Hypothesis

P028 and P029 both reduce the fraction of wrong-key valid-value swaps while
collapsing total retrieval. They show that suppressing one error category is
not enough when the recurrent map no longer learns usable values and reads.
The shared structural issue is that extra paired writes or sequential learned
transforms perturb both retention and storage at once.

Standard GDN2 also uses one normalized key for two different jobs: it chooses
the subspace to erase and the address at which to add the new payload. The
falsifiable P030 hypothesis is that independently learned erase and write
directions can reduce binding interference if the live transition remains one
stable, end-to-end learned dense-memory update. This is a foundational
recurrence, so it is trained from scratch on the validated directional MQAR
L1024 wrong-key regime rather than judged by a 100-step Sudoku graft.

## 3. Mechanism

Keep GDN2's Q/K/V projections, short convolution, decay, channel-wise write
gate, output path, K32xV32 state and native cross-layer terminal FutureSeed.
Interpret the existing erase projection as a unit direction `r`, the existing
key projection as a unit write address `p`, and add only a bias-free
`beta_proj: D -> H`.

For `D=Diag(exp(g))`, `beta=sigmoid(beta_proj(x))`, and `u=w*v`, apply

```text
S_t = D S_(t-1)
    - beta (sqrt(D) r)(sqrt(D) r)^T S_(t-1)
    + p [beta u]^T.
```

This maps exactly to one pinned official `chunk_dplr_delta_rule` call with
`a=sqrt(D)r`, `b=-beta*sqrt(D)r`, `k=p`, and `v=beta*u`. Its transition matrix
is `sqrt(D)(I-beta rr^T)sqrt(D)`, which is symmetric positive semidefinite and
has spectral norm at most one in exact arithmetic when `0<=D<=I`, `0<=beta<=1`
and `||r||=1`. Q/P/R normalization and transition-factor construction use
FP32; activations are cast back for the official BF16 operator.

The D128/L2 model has exactly 662,608 parameters, 1,024 more than native GDN2.
It retains exactly 4,096 recurrent state values per layer, adds no persistent
state, and uses one scan per layer.

## 4. Novel Boundary

P017/P019 wrap the original recurrence with routing or write control. P021 and
P028 add a second write. P024 changes address features without changing the
transition. P029 applies two sequential learned delta transformations. P027
adds a dense inverse-information state and is prohibitively slow. P030 instead
uses one independently addressed erase, one additive write and one dense state
inside one official DPLR transition. It has no cache, selector, second bank,
second payload, inverse Gram, reverse scan, task rule or custom recurrence.

## 5. Registered Contract

Before training, the strict A100 contract must prove:

- CUDA index0 and the registered physical UUID;
- pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e` and complete transitive DPLR source hashes;
- two official DPLR layers and exactly two `ChunkDPLRDeltaRuleFunctionBackward` paths, with no fallback;
- explicit-step parity from both zero and nonzero incoming state for output
  and terminal state, plus finite nonzero gradients through
  Q/P/R/V/decay/write/beta, initial state and native FutureSeed;
- exact 662,608 parameters, 4,096 state values per layer, zero state/scan increment;
- head permutation equivariance, incoming-state dependency, independent P/R dependency;
- transition symmetry error <=0.005, spectral norm <=1.001 and minimum eigenvalue >=-0.005.

## 6. Registered Science And Cost Gates

Run exactly one candidate on directional MQAR L1024 with 10,000 train and
1,000 validation examples. Reuse the locked historical native-FutureSeed
reference and the exact current-runtime reference; do not repeat control.

The candidate passes only if all conditions hold:

- two active layers and one active native FutureSeed route;
- mean `1-|p^T r| >=0.05` in each layer;
- finite nonzero beta, beta token/batch std >=`1e-4`, beta-weight RMS movement
  from initialization >=`1e-5`, erase strength, write RMS and board-varying
  terminal state;
- transition spectral norm <=1.001, minimum eigenvalue >=-0.005, symmetry error <=0.005, and terminal-state RMS in `[1e-4,1e4]`;
- balanced/future/past accuracy each >=0.85 and joint exact >=0.60;
- balanced accuracy >= historical `0.7475 + 0.10` and >= current-runtime control +0.10;
- wrong-key swap fraction at least 0.10 below both historical and locked
  runtime-control references, and total errors below both references;
- fit, post-warm wall and warmed-step ratios <=1.75x; peak allocation <=1.50x current-runtime control.

The cost denominator is the locked same-SKU A100 runtime control, not the same
physical card. These preregistered ratios remain conservative hard filters,
but are not interpreted as a paired latency estimate. Formal wall budget is
fixed at 2,700 seconds, matching the ledger's <=45-minute allowance.

Any contract, activation, stability, quality or cost miss closes this exact
equation. There is no angle, beta, rank, normalization, state, kernel, seed,
LR, loss, batch, epoch, width, depth, duration or Sudoku rescue.

## 7. Results

R1 reached the full production forward and both official DPLR backward paths,
then stopped in the checker before any science run. The checker assumed every
official projection exposed a direct `.weight`; `f_proj` is a composite module.
This is a non-science validation-harness failure. R2 aggregates finite nonzero
gradients over every trainable parameter in each projection module and adds
the already registered zero-state explicit recurrence check. It also makes
the activation interpretation executable with fixed beta-variation and
trained-weight-movement floors. The mechanism, data and quality gates are
unchanged.

R2 then passed the executable CUDA contract. It verified the target A100 UUID,
the registered DPLR source subtree, two official
`ChunkDPLRDeltaRuleFunctionBackward` paths, zero- and nonzero-incoming-state
explicit recurrence parity, every projection parameter gradient, the native
FutureSeed gradient, exact factor mapping, P/R dependency, head equivariance,
662,608 parameters, unchanged 4,096-value state, and no fallback.

All registered activation, sampled stability, and cost gates passed. Per-layer
beta mean/std was `0.3250/0.1983` and `0.2789/0.3145`; beta-weight RMS movement
was `0.02692/0.02547`. Erase/write separation was `0.8983/0.8607`; maximum
sampled transition spectral norm was `0.997874/0.999988`, minimum eigenvalue
`0.003977/0.000073`, and terminal-state RMS `0.2476/0.4699`. The independent
directions therefore learned and the sampled state geometry stayed bounded.

Quality nevertheless collapsed. Balanced/future/past/joint accuracy was
`0.0205/0.0180/0.0230/0`, versus the locked current native-FutureSeed
reference's `0.30625/0.3115/0.3010/0` and historical balanced `0.7475`.
Total errors rose from current `2,775` to `3,918`. Wrong-key valid-value swap
fraction fell from `0.458018` to `0.041858`, but only because nearly all useful
retrieval disappeared. Fit/post-warm/warmed-step/allocation ratios were
`1.1959/1.1971/1.3754/1.1767`, all within their registered ceilings. The
rejection is a clean quality result, not a cost or sampled-stability proxy.

A post-result review found two positive-claim limitations: the checker hashes
the DPLR subtree rather than every transitive shared kernel source, and it does
not assert the exact layer-0-terminal to layer-1-initial FutureSeed tensor
wiring. These omissions cannot turn the large quality rejection into a pass,
but this run must not support a positive provenance or wiring claim. The
endpoint's sampled spectral checks likewise support bounded observed geometry,
not a global empirical proof beyond the analytical factorization.

## 8. Decision

Reject P-GDN3-030 and close this exact separate-erase/write contractive-DPLR
equation. Do not transfer it to Sudoku or rescue angle, beta, normalization,
rank, state, kernel, seed, LR, loss, batch, epoch, width, depth, or duration.
Stable independently addressed erase and write are not sufficient to preserve
a learnable value/read map. Return to the registered receiver-native
FutureSeed formation question, where surprise has positive causal evidence
but sparse sequential replay has failed binding.

## 9. Provenance

- Run: `p-gdn3-030-contractive-dplr-l1024-r2-20260813T034209Z-b8e9342`
- Source SHA: `b8e93424efb108ef8b0f2d0e58eb2975860446ae`
- Contract/decision/checkpoint SHA256:
  `f07669aa3f711653069f918dcc9307a7baa53a994420aa52bfc05ab871840d34`,
  `0f5ff96519487d3f75b3b773e676a3d7944564b23ed35e0275042d4345369731`,
  `fd72b70e47d272e4ee88a53ddb59fab21345c2ba1c2204a1c1fbb86a701d6d06`.
- Candidate config/score/cases SHA256:
  `37a5274b9236ff843bef5f09dba9a340e285523854cd6f16e42b9235f3f4ec01`,
  `cb0bf5ccc6a04a9c216d77c80586bae0addae501cf28739156adf47d7265c961`,
  `d79b9a29126af4e78fb8395af37997cf9cb5faed22c591eca25cd79c62d042a2`.
- Formal-log/source-snapshot SHA256:
  `a4142e8347aefa62843afeec4fa5041735cb36883a83281e2e9a01146f51b7e9`,
  `13160b7f226c2500426467bedb3104e4d1022f286f56d934a77cc3563e0ef338`.
