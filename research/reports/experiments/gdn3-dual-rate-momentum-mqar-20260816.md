# P-GDN3-070: Coupled Dual-Rate Momentum

## 1. Question

Can one shared committed edit feed two stable Momentum time scales so that
their difference preserves recent owner identity without replacing P059's
successful global address, read geometry or slow integration?

## 2. Mechanism

Keep exact P059 Q/K/V width-four convolution, normalized shared key, residual,
all learned decay/write gates, `q-exp(D)k` output correction and native
FutureSeed. Replace the single velocity with coupled slow and fast lanes:

```text
r_t   = v_t - k_t^T(alpha_t S_{t-1})
u_t   = eta_t r_t
Mslow = mu_t Mslow_prev - k_t u_t^T
Mfast = mu_t^2 Mfast_prev - k_t u_t^T
B_t   = Mslow + tanh(W_lambda h_t) (Mfast - Mslow)
S_t   = alpha_t S_{t-1} - beta_t B_t
o_t   = q_t^T S_t
```

`W_lambda` is one zero-initialized D128-to-H4 weight per layer: exactly 1,024
new parameters for L2. Persistent inference state changes from `[S,M]` to
`[S,Mslow,Mfast]`, adding exactly 4,096 FP32 values per layer. There is still
one causal token scan, one read state, one address and one committed edit.
Native FutureSeed transports all three state planes through the existing one
gate per receiver.

The clean-room Triton prototype stores one FP32 three-plane checkpoint every
eight tokens for stable backward recomputation. This is a training-memory
implementation choice; inference remains constant-state and linear-time.

## 3. Why Existing Failures Do Not Cover It

- P063/P067 and the direct `decouple_k` family introduced incompatible
  addresses. P070 shares the exact parent address and edit.
- P053/P064 and P-GDN3-006 added independent banks or experts. P070 has one
  read state and two coupled velocities, not averaged outputs.
- P068 replaced global Momentum with a current-key projection. P070 preserves
  the complete slow P059 path at zero mix.
- P062 changed where the residual is evaluated and used an unstable full
  reverse reconstruction. P070 keeps the parent residual and bounds reverse
  recomputation to eight-token blocks.
- P-DIAG-MOMREAD-001 proves `q-Dk` is necessary; P070 leaves it unchanged.

## 4. Falsifiable Prediction

If P059's 151 adjacent-owner swaps are caused by one slow velocity blending
neighboring commits, the learned band-pass `(Mfast-Mslow)` should activate with
board/token variation, retain both directional accuracies, improve balanced
accuracy by at least `.005`, and remove at least 25 wrong-key swaps. If it
cannot do so in the established directional MQAR L1024 binding regime, close
dual-rate Momentum; do not tune the fast exponent, mix scale, checkpoint
length, sharing, seed, LR, loss, width, depth or duration.

## 5. Frozen Protocol

- candidate only, trained from scratch on directional MQAR L1024;
- D128/L2/H4/K32/V32, four KV pairs, 10 epochs, batch32, seed123;
- exact P059 matched initialization, train/test data and frozen score/cases;
- exact external Momentum DeltaNet SHA `c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`;
- one registered A800, CUDA index 0, no concurrent model/eval;
- no Sudoku transfer unless every gate passes; no sweep or rescue.

## 6. CUDA Contract

Before formal training require:

1. exact GPU UUID, source SHA/readback, external Momentum and host FLA hashes;
2. clean-room Triton/Torch output, three-state and all-gradient parity;
3. zero-mix output and first two state planes match native P059 for arbitrary
   nonzero incoming state within `.02` relative RMS;
4. exactly two custom backwards, +1,024 parameters and +4,096 persistent state
   values per layer, all new weights exactly zero before the probe;
5. Q/K/V/alpha/mu/beta/eta, both mix weights and the native FutureSeed edge
   receive finite nonzero gradients;
6. head equivariance, causal-prefix identity, explicit fast-state dependence,
   finite state, and active mix after one deterministic gradient-direction
   probe.

Any miss is an integrity failure; do not launch formal training.

## 7. Formal Gates

Integrity, all required:

- exact external source, parent mapping, data and warmup batch;
- exactly 600,696 parameters and 12,288 recurrent values per layer;
- exactly one clean-room checkpointed scan and native FutureSeed.

Activation/stability, all required in both layers:

- band/slow relative RMS at least `.01`;
- injected band relative RMS at least `1e-4` with nonzero board/token variation;
- mean absolute mix at least `1e-4`, maximum at most `.95`;
- S, slow M and fast M finite and nonzero; slow/S RMS in `[.01,20]`.

Quality versus frozen P059, all required:

- balanced accuracy gains at least `.005`;
- future and past each regress by at most `.005`;
- joint exact gains at least `.005`;
- total errors fall by at least 20;
- wrong-key swaps fall by at least 25;
- adjacent swaps fall by at least 20 percent.

Prototype cost versus P059, all required:

- elapsed, post-warm wall and independently warmed step each below `3x`;
- peak allocation below `1.6x`.

## 8. Decision

Pass admits one Sudoku scaffold transfer and a fused/chunk production kernel.
Any quality, activation, stability or cost miss closes this exact dual-rate
state organization without rescue. The next branch must use the paired error
transitions to choose either a different scalable live recurrence or a
receiver-native FutureSeed mechanism; it must not revisit output-read, direct
key decoupling or tail-prefill families.

## 9. Status

Approved and implemented; awaiting exact pushed-source CUDA contract.
