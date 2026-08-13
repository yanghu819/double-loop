# P-GDN3-030: Contractive Separate-Erase/Write DPLR

## 1. Metainfo

- Status: preregistered; implementation pending strict CUDA contract
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
- explicit-step parity for zero and nonzero incoming state, output and terminal state;
- finite nonzero gradients through Q/P/R/V/decay/write/beta, initial state and native FutureSeed;
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
- finite nonzero beta, erase strength, write RMS and board-varying terminal state;
- transition spectral norm <=1.001, minimum eigenvalue >=-0.005, symmetry error <=0.005, and terminal-state RMS in `[1e-4,1e4]`;
- balanced/future/past accuracy each >=0.85 and joint exact >=0.60;
- balanced accuracy >= historical `0.7475 + 0.10` and >= current-runtime control +0.10;
- wrong-key swap fraction at least 0.10 below historical and total errors below both references;
- fit, post-warm wall and warmed-step ratios <=1.75x; peak allocation <=1.50x current-runtime control.

Any contract, activation, stability, quality or cost miss closes this exact
equation. There is no angle, beta, rank, normalization, state, kernel, seed,
LR, loss, batch, epoch, width, depth, duration or Sudoku rescue.

## 7. Results

Pending.

## 8. Decision

Pending the fixed L1024 gate. Only a complete pass permits one matched hard
Sudoku transfer. A failure returns the program to the preregistered
receiver-native FutureSeed formation question, not a nearby P030 sweep.

## 9. Provenance

Exact source SHA, worktree, contract, run, config, score, cases, checkpoint,
logs and SHA256 values will be appended after execution.
