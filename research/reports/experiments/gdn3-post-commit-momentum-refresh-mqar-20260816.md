# P-GDN3-061: Post-Commit Momentum Refresh

## 1. Metainfo

- Status: closed at the strict CUDA contract; no formal training
- Decision field: directional MQAR L1024 wrong-owner tail
- Candidate: D128/L2/H4/K32/V32 Momentum Delta + native `[S,M]` FutureSeed
- Fixed training: 10 epochs, batch32, seed123, exact P059 data and initialization
- Frozen reference: P-GDN3-059 balanced/future/past/joint
  `.94425/.95150/.93700/.82400`, 223 errors and 151 wrong-key swaps
- Added parameters/persistent state: exactly zero/zero

## 2. Evidence And Hypothesis

P059 proves that second-order Momentum plus native `[S,M]` FutureSeed is the
first architecture to close most of the L1024 ownership problem. Component
ablation then shows that `M`, not `S`, carries almost all useful cross-layer
future evidence. P-DIAG-MOMQCF-001 rules out a simple receiver-key mismatch:
forcing the final layer to query with the exact key from its own write token
repairs only `14/151` swaps and collapses future accuracy
`.95150 -> .04700` while preserving past accuracy `.93700 -> .93350`.

The remaining falsifiable hypothesis is that the native Momentum update uses
the pre-commit residual to form `M`, then changes `S` without refreshing that
residual. The next token and the next layer therefore inherit stale
post-commit error in the very component FutureSeed actually transports.

## 3. Mechanism

For normalized K, the unchanged parent microstep is

`r_t = v_t - alpha_t k_t^T S_{t-1}`,

`M_t = mu_t M_{t-1} - eta_t k_t r_t^T`,

`S_t = alpha_t S_{t-1} - beta_t M_t`.

Immediately after it, append exactly one refresh microstep using the same
`k_t`, `v_t` and `eta_t`, with `log_alpha=0`, `log_mu=0` and `beta=0`:

`rho_t = v_t - k_t^T S_t`,

`M_t+ = M_t - eta_t k_t rho_t^T`,

`S_t+ = S_t`.

Only the parent/even output is returned. The next logical token starts from
`[S_t+, M_t+]`, and native FutureSeed transports the final refreshed pair.
The external operation receives one interleaved length-`2T` sequence, so
chunk training and its backward are retained. The refresh adds no learned
gate, address, scan state, cache or task-specific logic.

Fixing `mu=1` on the refresh is essential: when `rho_t=0`, both `S` and `M`
are unchanged. Reusing the parent decay would confound residual correction
with a doubled momentum time constant and is not authorized.

## 4. Why Existing Failures Do Not Cover It

- P021 performs a second independently learned write that changes the primary
  state twice; P061's refresh cannot change `S` and has no new projections.
- P037 replaces the parent GDN2 edit with one scalar committed residual; P061
  preserves the complete successful Momentum transition and only refreshes
  its post-commit error component.
- P060 and the decoupled-key family change address geometry. P061 keeps Q/K
  and all learned gates exact.
- P005-P008, replay/cache, Raven controllers and side memories act outside the
  token-local Momentum commit. P061 modifies the live recurrent transition
  that MOMQCF specifically opened.

## 5. Strict CUDA Contract

The sole-GPU contract must prove all of the following before formal training:

1. exact GPU UUID, external Momentum SHA `c6e77fa`, host FLA SHA `9c8e42e`,
   source hashes and two external Momentum chunk backward paths;
2. exact P059 initialization hash
   `dc8f49f92c00f0f08863cf6183692246cf2c44f0c5aa3a35892bb9555ec17bfb`
   and shared-parent hash
   `0adf26f657e59a35e90d7e54b905721b10f537b6391d107676248de16b71f13f`;
3. zero parameter and persistent-state delta versus P059;
4. exactly one refresh after every logical token in both layers, `T -> 2T`,
   parent outputs selected from even positions and `cu_seqlens=None`;
5. refresh `alpha=1`, `mu=1`, `beta=0`, same K/V/eta, exact unchanged `S`,
   finite nonzero changed `M` when post-commit residual is nonzero, and
   fixed-point identity when that residual is zero;
6. parity with an explicit sequential parent-then-refresh recurrence,
   including nonzero incoming `[S,M]`;
7. finite nonzero gradients to Q/K/V/alpha/beta/mu/eta, active native
   FutureSeed, bounded finite state and no fallback.

Any miss is an integrity failure and closes the exact mechanism before formal
training. There is no refresh-count, order, coefficient or tolerance rescue.

## 6. Registered Prediction And Gates

Activation requires both layers to report exactly `1024` parent and `1024`
refresh tokens per evaluation sequence, refresh parent/odd read relative RMS
at most `1e-5`, finite state and Momentum RMS, and one active native
FutureSeed route. Synthetic post-commit Momentum change must be at least
`1e-4` relative RMS with nonzero batch/token variation.

Quality is measured against frozen P059 and passes only if every check holds:

- balanced accuracy at least `.95500` and gain at least `.01000`;
- joint exact at least `.84000` and gain at least `.01000`;
- future and past accuracy each no more than `.00500` below P059;
- total errors at most 180;
- wrong-key valid-value swaps at most 105;
- conditional wrong-key fraction at most `.60713`, a reduction of at least
  `.07000` from P059.

This is a deliberately decision-changing gate: a small CE movement without
closing the residual owner tail does not justify doubling scan work.

## 7. Cost And Kill Gate

Relative to frozen P059, elapsed, post-warm wall and independently warmed-step
ratios must each be below `2.25x`; peak allocation must be below `1.20x`.
Transient token work is exactly doubled, while parameters and persistent state
remain unchanged.

Any integrity, activation, quality or cost miss closes Post-Commit Momentum
Refresh. Do not rescue refresh decay, residual coefficient, count, order,
address, gate, seed, LR, loss, batch, width, depth, duration or initialization.
Only a full pass authorizes one matched hard-Sudoku transfer.

## 8. Required Artifacts

Archive contract and formal score, frozen-reference hashes, candidate cases,
checkpoint, validation curve, transition taxonomy, refresh/state diagnostics,
throughput, memory, GPU samples, source/config/log hashes and GitHub readback.

## 9. Result And Decision

Exact pushed/read-back source
`832f5258512fe8cf8c8665cb2dbb5b2eb9e0963d` ran from a clean detached
worktree on A800 index0 UUID
`GPU-c1d7c624-a393-befa-3807-7e00602d65ca`. The implementation reached the
full-stack GPU contract with two exact refresh mixers, the fixed `1024->2048`
microstep schedule and zero refresh log-alpha, log-mu and beta.

The contract then failed the preregistered state-read identity before formal
training. Parent-versus-refresh output relative RMS was
`2.5927345e-4/2.6428906e-4` across the two layers, with maximum BF16 changes
`2.44140625e-4/9.765625e-4`. Both exceed the frozen `1e-5` ceiling. This is a
small numerical discrepancy, but the gate intentionally required the M-only
microstep to leave the primary-state read invariant under the production
chunk path. The miss is binding and is not reinterpreted after observation.

Close P-GDN3-061 without tolerance, output probe, refresh coefficient, decay,
count, order, seed, LR, loss, batch, width, depth or duration rescue. No
checkpoint or quality score exists and no Sudoku transfer is authorized.

Artifacts:

- run:
  `/huyang2/double-loop/runs/p-gdn3-061-momentum-refresh-20260816T013400Z-832f525`;
- contract log SHA256:
  `22fb53bd3152e39b4770f36b643bf96ed7a7b6e978f04152412d0f335c4d67a0`;
- abort SHA256:
  `b9870ffdacac54be4b55be1b193b9433fe98819e68c628656d058ddd5fad56c3`;
- GPU samples SHA256:
  `cdb55c17be7e6847839dc25c5f857e995d8bc8234a30af5693d0b51630c48d6e`;
- source snapshot SHA256:
  `50b83cae8b81526878623049512b5e139957434b02a31daf885ddb59514eb9bc`.

There were 32 sampler rows and 13 active-memory rows. Active utilization
averaged `39.69%`, peaked at `61%`, peak sampled memory was `3,000 MiB`, and
peak sampled power was `199.90 W`. The A800 returned to 0%/0 MiB with no
compute process after the exact contract failure.
