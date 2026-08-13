# P-FS2-008: Receiver-Native Surprise-Weighted Ridge-Residual FutureSeed

## 1. Metainfo

- Status: proposed; preregistered; implementation and CUDA contract pending
- Decision field: directional MQAR L1024 wrong-key binding regime
- Fixed setting: D128/L2/H4/K32/V32, 10 epochs, batch32, seed123,
  10,000/1,000 examples
- Fixed arms: contemporaneous native FutureSeed control, then one
  surprise-weighted ridge-residual candidate
- New trainable parameters: zero
- Persistent recurrent-state delta: zero
- Main recurrent scans: unchanged, exactly one pinned official GDN2 scan per
  layer
- First transfer target after a complete pass: hard 9x9 Sudoku 51-64 blanks

## 2. Evidence And Hypothesis

P-FS2-007 established that exact committed-edit surprise is causal but sparse
replay is the wrong state-construction operator. Its surprise-K16 arm improved
balanced accuracy from the contemporaneous native FutureSeed control's
`0.30625` to `0.48825`, and beat matched recency-K16 by `0.47300`. However,
`2,044/2,047 = 0.998534` of its remaining query errors were correct values
retrieved from the wrong key. The sparse tape retained useful value evidence
while its K16 sequential replay rebuilt a badly colliding address map.

The falsifiable P-FS2-008 hypothesis is that native FutureSeed preserves useful
coarse memory but lacks a receiver-native correction for address collisions.
FutureSeed should therefore retain its exact native seed and add a bounded
receiver-native binding residual over all canonical token evidence, with actual
committed-edit magnitude used only as a continuous importance weight. The
receiver must form keys and values with its exact causal-convolution
preprocessing, then fit only the payload error left by the native seed. This
removes sparse admission and replay-order loss without transporting producer
K/V coordinates or replacing the native memory.

The mechanism passes only if that bounded complement converts the P-FS2-007
value-retention signal into substantially better key-value binding. A raw
ridge-fit improvement without lower same-weight counterfactual query CE,
endpoint accuracy and wrong-key swaps falsifies the hypothesis.

## 3. Exact Mechanism

The producer's pinned official GDN2 scan is unchanged. For producer token `t`,
let `S_(t-1)^P` be the live state before the token update and let
`D_t^P = Diag(exp(g_t^P))`. Capture the actual value residual committed by the
official recurrence:

```text
e_t^P = w_t^P * v_t^P
      - (b_t^P * k_t^P)^T [D_t^P S_(t-1)^P].
```

The capture must be the exact tensor already produced inside the official
forward, not a second recurrence or an approximation. Define one detached
board-level score per token and normalize it to mean one:

```text
s_t     = ||e_t^P||_F                         over all H and V channels
omega_t = L * s_t / sum_j s_j
sum_t omega_t = L.
```

`sum_j s_j` must be finite and strictly positive. Otherwise the candidate
fails immediately. `s` and `omega` are stop-gradient evidence, so surprise is
not a learned selector.

Let `x_t^P` be the differentiable, producer-completed residual stream delivered
to the receiving layer, retaining canonical token and position evidence. The
surprise score remains detached, but `x^P` does not. Using a fresh zero
convolution state and exactly the same sequence boundaries, activation,
parameters and causal `ShortConvolution` implementation as its normal main
call, the receiver forms finalized tuples:

```text
kbar^R = KConv^R(K_proj^R(x^P); fresh_zero_cache)
vbar^R = VConv^R(V_proj^R(x^P); fresh_zero_cache)
kappa_t^R = l2_normalize(kbar_t^R)                    in R^K
u_t^R     = sigmoid(W_proj^R(x_t^P)) * vbar_t^R       in R^V.
```

No convolution cache or projected tensor is borrowed from the producer. The
extra receiver K/V preprocessing is causal and parameter-shared with the
receiver main path, but it is not a recurrent scan. Token-order invariance is
claimed only for aggregation of the finalized `(kappa_t^R, u_t^R, omega_t)`
tuples, after causal preprocessing.

First construct the exact native injected seed with the existing receiver
FutureSeed operator:

```text
Z_0^R = make_initial_state^R(S_T^P),
r_t^R = u_t^R - (kappa_t^R)^T Z_0^R.
```

`S_T^P` is the producer terminal state used by native FutureSeed. For each
board and head, solve only for the residual left by that native seed:

```text
Delta_raw = argmin_Delta sum_t omega_t
              ||(kappa_t^R)^T Delta - r_t^R||_2^2
              + lambda ||Delta||_F^2,

G_R = sum_t omega_t kappa_t^R (kappa_t^R)^T,
C_R = sum_t omega_t kappa_t^R (r_t^R)^T,
lambda = 2^-8 * (trace(G_R) / K + 1e-6),
Delta_raw = (G_R + lambda I)^-1 C_R.
```

`lambda` is detached and fixed by this formula. Form `G_R`, `C_R`, `lambda`,
the Cholesky factor and the solve in FP32 with autocast disabled. Symmetrize
`G_R` once as `0.5 * (G_R + G_R^T)`, call
`torch.linalg.cholesky_ex`, require `info == 0` for every board/head, and use
`torch.cholesky_solve`. There is no jitter loop, pseudoinverse, least-squares
fallback or precision fallback. The `2^-8` coefficient is fixed before the run;
for unit keys and weights summing to L it bounds the worst rank-one regularized
Gram condition number below `1e4`.

Bound the solved correction in the already-injected receiver state coordinates,
independently per board and head:

```text
q_R     = min(1, RMS(Z_0^R) / (RMS(Delta_raw) + 1e-6)),
Delta_R = q_R * Delta_raw,
Z_R     = Z_0^R + Delta_R.
```

`q_R` is an algebraic, non-learned stability bound. It guarantees
`RMS(Delta_R) <= RMS(Z_0^R)` up to the registered epsilon and therefore bounds
candidate state RMS by `2x` native under the triangle inequality. There is no
second FutureSeed normalization after addition: `Z_0^R` is the exact native
injected seed and `Delta_R` is its bounded receiver-coordinate complement.
The ridge statistics are rebuilt for each producer-to-receiver edge and each
forward. Nothing survives a forward or macro-loop boundary.

There is no `topk`, sort, gather, truncation, event replay, event cache,
producer-basis K/V/state transfer, second scan, second recurrent bank, learned
admission rule or task-specific logic.

## 4. Exact Tensor And Cost Accounting

For the fixed D128/L2/H4/K32/V32, L1024 experiment, the sole cross-layer edge
uses:

| Quantity | Exact shape | Role |
| --- | --- | --- |
| canonical producer evidence `x^P` | `[B,1024,128]` | differentiable producer-completed token/position evidence |
| committed edit `e^P` | `[B,1024,4,32]` | exact official value residual |
| surprise score `s` | `[B,1024]` | detached token importance |
| normalized weight `omega` | `[B,1024]` | positive weights summing to 1024 |
| receiver key `kappa^R` | `[B,1024,4,32]` | exact receiver K-projection plus causal-conv unit address |
| receiver payload `u^R` | `[B,1024,4,32]` | exact receiver V causal-conv times native write gate |
| native injected seed `Z_0^R` | `[B,4,32,32]` | unchanged native FutureSeed state |
| native read residual `r^R` | `[B,1024,4,32]` | payload error left by native seed |
| weighted Gram `G_R` | `[B,4,32,32]` | receiver address geometry |
| cross-covariance `C_R` | `[B,4,32,32]` | receiver key-value binding |
| ridge scale `lambda` | `[B,4,1,1]` | fixed detached regularization |
| raw ridge residual `Delta_raw` | `[B,4,32,32]` | receiver-native residual solve |
| bounded residual `Delta_R` | `[B,4,32,32]` | per-board/head RMS-bounded complement |
| injected state `Z_R` | `[B,4,32,32]` | native seed plus bounded complement |

The control and candidate must each have exactly `661,584` trainable
parameters and `4,096` persistent recurrent-state values per layer. Candidate
parameter names, values, buffers and initialization hashes must exactly match
the control. The candidate adds zero parameters, zero persistent state and
zero recurrent scans. Its only training-path extra work is fresh receiver K/V
causal-convolution preprocessing, residual construction, two FP32 weighted
reductions and one batched K32 Cholesky solve per board/head on the single layer
edge. Expensive prediction contractions, condition estimates and native
counterfactual forward diagnostics are disabled during training and run only on
the fixed endpoint evaluation batch. No `[L,L]` tensor may be materialized.

## 5. Novelty Boundary

- P-FS2-007 selects K16 events and sequentially replays them through a receiver
  scan. P-FS2-008 uses all L1024 finalized causal tuples and directly solves one
  order-independent residual; it has no sparse tape and no replay scan.
- Native FutureSeed transports the producer terminal KxV state. P-FS2-008 keeps
  that exact native seed and transports only canonical hidden evidence plus a
  scalar importance field for a receiver-native residual complement.
- P-FS3 producer codecs and routers transform or compress producer-basis state.
  P-FS2-008 never reads producer K/V coordinates when constructing `Delta_R`.
- Dense inverse-Gram recurrent experiments changed the live token recurrence
  and carried extra matrix state. Here the main recurrence and persistent
  state are unchanged; the solve is a bounded per-edge FutureSeed formation
  operator.
- Surprise is the exact committed edit norm, not a learned selector, Sudoku
  heuristic, recency prior, serving-time prompt cache or HYPIC/PIC reuse.

## 6. Strict CUDA Contract

Before science, the exact pushed SHA in a clean detached worktree must prove:

1. one visible AIStation GPU at CUDA index0 with the registered physical UUID;
2. pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e`,
   exact GDN2 source hashes, Triton short convolution, fixed data hashes and no
   backend fallback;
3. candidate and control have identical parameters, names, tensors, buffers,
   initialization hashes, optimizer inputs and exact `661,584` parameter
   count;
4. committed-edit capture has exact output, terminal-state and captured-edit
   parity for zero and nonzero incoming states, and does not execute a second
   recurrence;
5. the full L2 candidate backward contains exactly two
   `ChunkGDN2FunctionBackward` paths, with exactly one official main scan per
   layer and no replay or auxiliary scan;
6. source and runtime wiring contain no Top-K, sorting, event gather, event
   tape/cache, producer K/V payload, second bank or silent fallback;
7. candidate K/V tuples use the receiver's exact K/V projections and causal
   `ShortConvolution` weights with fresh zero cache; direct pointwise bypass,
   producer convolution cache reuse and producer K/V coordinates are absent;
8. `omega` is finite, nonnegative, detached and sums to exactly L within
   `1e-5` relative error for every board; the score path has no gradient, while
   producer-completed evidence, receiver K/V/W projections and the native
   FutureSeed gate receive finite nonzero gradients through the residual seed;
9. jointly permuting finalized `(kappa^R,u^R,omega)` tuples changes
   `Delta_raw` by at most `1e-5` relative error; raw-token permutation before
   causal convolution is explicitly not required to be invariant;
10. with finalized evidence and weights fixed, perturbing receiver K/V/W
    projections changes `Delta_R`, while perturbing producer K/V coordinates
    does not, proving receiver-native formation and absence of producer-basis
    transfer;
11. FP32 `cholesky_ex` succeeds for 100% of board/head systems, every
    regularized system has condition number `<=1e4`, and
    `||(G_R + lambda I)Delta_raw - C_R||_F / (||C_R||_F + 1e-8) <=1e-4`;
12. `Z_0^R`, `Delta_raw`, `Delta_R` and `Z_R` are finite with exact
    `[B,4,32,32]` shape; `RMS(Delta_R) <= RMS(Z_0^R) + 1e-6` per board/head,
    candidate state RMS is no more than `2x` native, and board variation is
    finite and nonzero;
13. training mode does not materialize read predictions, condition spectra or
    counterfactual logits; these diagnostics activate only under the registered
    endpoint-evaluation flag; and
14. no CPU model smoke, concurrent GPU model/eval process or unregistered
    artifact mutation occurs.

Any contract or integrity miss closes this exact implementation before
science. It does not authorize a numerical fallback or a mechanism change.

## 7. Fixed Matched Protocol

Run exactly two arms in fixed order on directional MQAR L1024: a
contemporaneous native FutureSeed control and one surprise-weighted ridge-residual
candidate. Both use D128/L2/H4/K32/V32, 10 epochs, batch32, seed123 and the same
10,000/1,000 examples. Reset RNG before each arm and require identical data,
initialization, optimizer, schedule, batch-order and first-anchor hashes.

The SHA-locked historical native FutureSeed endpoint remains the absolute
reference: balanced `0.7475` and wrong-key valid-value swap fraction
`0.8069306931`. The same-runtime native arm is the paired quality, error and
cost denominator. P-FS2-007 surprise-K16 balanced `0.48825` and
`2,044/2,047` wrong-key swaps are mechanistic evidence, not a relaxed quality
reference.

On one fixed held-out activation batch, first report raw weighted ridge
write-fit MSE as a numerical diagnostic only. Then evaluate the full fixed test
set twice through the unchanged recurrence and output head using the same
trained candidate checkpoint, data hash, case IDs, labels, query positions and
targets. The only changed mechanism is the initial state: `Z_R` for the
candidate evaluation and `Z_0^R` for the native counterfactual. Define

```text
query_CE_candidate = CE(logits(x, initial_state=Z_R)[query_mask], labels),
query_CE_native_cf = CE(logits(x, initial_state=Z_0^R)[query_mask], labels).
```

This is a same-weight native counterfactual: it does not compare separately
trained parameter sets and it cannot affect training or reported primary
metrics. It tests whether the residual itself improves actual labeled receiver
queries rather than only its closed-form proxy objective.

## 8. Registered Activation, Science And Cost Gates

The candidate passes only if every applicable condition below holds.

Activation and numerical gates:

- exact committed-edit score is finite, positive and token-varying;
- receiver causal-conv keys/payloads, residual and injected seed have finite nonzero
  board and token variation;
- Cholesky succeeds for every board/head and regularized condition number is
  `<=1e4`;
- raw weighted write-fit MSE is finite and at most `0.75x` the zero-residual
  native proxy on the fixed activation batch;
- same-weight candidate pooled query CE is at most `0.98x` the
  native-counterfactual query CE, and each of future and past query CE is
  strictly lower; and
- bounded residual RMS is no greater than native seed RMS per board/head, while
  candidate state RMS remains no more than `2x` native.

Quality gates:

- balanced accuracy `>=0.85`;
- future accuracy `>=0.83`;
- past accuracy `>=0.83`;
- joint exact `>=0.60`;
- balanced accuracy at least `+0.10` over historical `0.7475`;
- balanced accuracy at least `+0.20` over the contemporaneous native control;
- wrong-key valid-value swap fraction at least `0.10` below historical
  `0.8069306931` and at least `0.10` below the contemporaneous native control;
  and
- total query errors strictly below both the locked historical reference and
  the contemporaneous native control.

Cost gates relative to the contemporaneous native control:

- fit elapsed ratio `<=1.60x`;
- post-warm wall ratio `<=1.60x`;
- independently warmed training-step ratio `<=1.60x`; and
- peak CUDA allocation ratio `<=1.25x`.

The absolute balanced threshold and historical `+0.10` condition are both
registered even though `0.85` is slightly stronger than `0.8475`. Neither may
be silently dropped.

## 9. Kill Rule And Next Decision

Any contract, activation, numerical, quality, binding, integrity or cost miss
closes the receiver-native surprise-weighted ridge-residual family. There is no rescue
by changing ridge coefficient, jitter, solver, precision, score transform,
weight exponent, temperature, residual bound, clipping, normalization, Top-K,
replay order, event count, receiver convolution/projection, FutureSeed scale, seed, LR, loss, batch,
epoch, duration, width or depth.

A full pass authorizes one fixed uniform-weight attribution control to test
whether committed-edit surprise, rather than receiver-native ridge formation
alone, supplies the gain. Only retained performance after that attribution
permits a D256/L12 hard-Sudoku transfer and later fused implementation. A miss
returns the program to live recurrent address-binding dynamics rather than
another cache, replay or capacity variant.

Exact source SHA, run names, GPU identity, contract, score, cases, checkpoint,
timing, memory and artifact SHA256 values remain pending and must be appended
only after execution.
