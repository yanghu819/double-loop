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

Completed and discarded. Exact pushed/read-back source
`11372b96913bb930be08a6a22dedfec44e147410` ran from clean detached worktree
`/huyang2/double-loop/worktrees/p-fs2-015-8705a09` on A800 CUDA index 0, UUID
`GPU-c1d7c624-a393-befa-3807-7e00602d65ca`.

The strict CUDA contract passed. It verified the exact external Momentum SHA,
one clean-room fused checkpointed scan, output/state/all-gradient parity near
`1e-6`, zero-mix P059 output parity and first-two-state parity at
`1.22e-7`, exact +1,024 parameters and +4,096 recurrent values/layer,
causality, head equivariance, explicit fast-state dependence, full-stack
gradients and active bounded synthetic dual-rate geometry. Contract JSON
SHA256 is `ba3f18c4d52e35e2a282411e4871a449b83e527d3575d2ed9cf846cf0f63e917`.

The sole formal endpoint completed ten epochs and failed decisively:

| Metric | P059 control | P070 candidate | Delta |
|---|---:|---:|---:|
| balanced accuracy | .94425 | .00850 | -.93575 |
| future accuracy | .95150 | .00950 | -.94200 |
| past accuracy | .93700 | .00750 | -.92950 |
| joint exact | .82400 | 0 | -.82400 |
| total errors | 223 | 3,966 | +3,743 |
| wrong-key valid-value swaps | 151 | 109 | -42 |
| adjacent-owner swaps | 151 | 53 | -98 |

The lower swap counts are a collapse artifact. Of 3,777 parent-correct
queries, 3,645 become unrelated wrong values and another 99 become wrong-key
errors. Only one old wrong-key error becomes correct. Validation remains near
chance for every epoch, so the added fast lane removes the sharp P059 learning
transition rather than cleaning its residual tail.

Both dual-rate lanes are live and finite. Layer 0/1 band-to-slow relative RMS
is `.40416/.40667`, injected-band relative RMS is `.58796/2.13602`, and mean
absolute token mix is `.09791/.59569`. Layer 1 reaches maximum absolute mix
`.96869`, above the registered `.95` ceiling. Slow/state RMS remains bounded,
so broad failure is not NaN, overflow, fallback or dead activation; a learned
fast-minus-slow component perturbs the optimization trajectory too strongly
before useful retrieval emerges.

Elapsed/post-warm/warmed-step/peak-allocation ratios are
`1.33560/1.34121/1.49917/1.40304x`, all inside the fixed cost limits. Across
71 five-second telemetry samples, 34 compute-active samples average `51.21%`
GPU utilization, peak at `66%`, and observed memory peaks at `2,178 MiB`.
Formal endpoint status is zero; wrapper exit status `2` denotes the registered
science-gate miss.

Discard P-GDN3-070. Close fast-decay exponent, mix scale or initialization,
controller sharing, checkpoint spacing, seed, LR, loss, batch, width, depth
and duration rescue. The useful boundary is stronger than "two time scales do
not help": they suppress adjacent-owner swaps while simultaneously preventing
the base retrieval transition. The next mechanism must preserve P059's exact
learnable path while adding owner evidence that cannot dominate early
training.

Artifacts:

- run: `/huyang2/double-loop/runs/p-gdn3-070-dual-rate-momentum-20260816T1355Z-11372b9`;
- comparison SHA256: `f57e048ee9e753d78c79be5d20b33b36eba24d5c4214454964a91712fecb62d4`;
- checkpoint SHA256: `b0e0fd8e0a09c9df46d1b433ba510d2b97ed78b5895b01701bbc976f9ed2b43d`;
- cases SHA256: `7649ada8f33364e629624c08f3eb88201b95cc6c9db705564b31425cbc3b60c6`;
- GPU samples SHA256: `324032dbeab3513bc7164ef6d86b01e10b91631e2a21ae2fa8590f8bfbc9b526`;
- source snapshot SHA256: `b4b006537e669db21b5f0a6ef9492fa8e747813a7482bf1d4b1183fbc06949fc`.
