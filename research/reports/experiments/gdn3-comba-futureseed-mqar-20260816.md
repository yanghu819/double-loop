# P-GDN3-065: Closed-Loop Comba FutureSeed

## 1. Research Question

Can a complete closed-loop committed-residual recurrence remove Momentum
DeltaNet's remaining wrong-owner errors without giving up native FutureSeed or
linear recurrent memory?

## 2. Mechanism Hypothesis

P059 is the strongest result: balanced/future/past/joint
`.94425/.95150/.93700/.82400`, with 223 errors and 151 wrong-key valid-value
swaps. P064 then expanded state capacity 16x with a primary sparse slot bank,
but future accuracy collapsed to `.02850`. Capacity alone is not the missing
ingredient.

Comba changes the live token transition. For each token it predicts the current
value from the pre-update state using prediction key `p_t`, computes the exact
residual, decays the state, and writes only that residual under `k_t`:

`r_t = v_t - p_t^T S_{t-1}`

`S_t = exp(g_t) S_{t-1} + beta_t k_t r_t^T`.

Here `p_t = sigmoid(decay_h) k_t`, so prediction and committed owner stay
collinear instead of learning two unrelated address frames. The output query
also receives the architecture's native correction `q_t - D_h p_t`. The
hypothesis is that a token should erase only what its own address predicts and
commit only the unresolved value, reducing valid-value/wrong-key swaps.

Native FutureSeed remains unchanged: layer 0's terminal `[B,H,K,V]` state is
RMS-normalized and seeded into layer 1 through one learned gate.

## 3. Fixed Implementation And Provenance

- Read-only external dependency:
  `HuuYuLong/MomentumDeltaNet@c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`.
- Exact Comba layer, chunk and fused-recurrent file hashes are asserted. The
  repository declares no license, so no external implementation is copied or
  redistributed here; this repository contains only a thin import adapter.
- Host FLA remains pinned at
  `9c8e42e762fce087c27b673af4922795d9edb85e` and supplies shared utilities.
- D128/L2/H4/K32/V32, chunk training, short convolution, output gate, native
  output correction and inner prediction decay. One K32xV32 state per layer,
  exactly 4,096 recurrent values.
- Directional MQAR L1024/K4, 10 epochs, batch32, seed123. Shared compatible
  parent tensors, data hashes, warmup batch and post-warm determinism match the
  frozen runs.
- Frozen comparator is P059 Momentum DeltaNet plus native `[S,M]` FutureSeed.
  It is not rerun.
- One candidate only. No correction-factor, decay, output-gate, inner-decay,
  seed, LR, loss, batch, width, depth or duration sweep.

## 4. Falsifiable Prediction

If the residual tail is caused by incoherent ownership during the live write,
the complete closed-loop transition should improve both retrieval exactness
and wrong-key swaps, not merely activate or lower conditional swap share by
creating broad errors. If it cannot beat P059 under the fixed gate, this exact
Comba transfer is closed.

## 5. Registered Gates

Integrity and activation:

- exactly one registered A800 at CUDA index 0;
- exact pushed clean detached source, external SHA and file hashes, frozen
  artifact hashes, data hashes and warmup batch;
- exactly two `ChunkCombaFunctionBackward` paths, finite nonzero Q/K/V,
  decay-logit, write-gate and output-correction gradients, plus the receiving
  FutureSeed gradient;
- exact parameter delta `-63,976` versus native GDN2 and exactly 4,096 state
  values per layer;
- chunk/fused-recurrent parity within `.05` relative RMS with a real nonzero
  initial-state dependency;
- two finite nonzero states with board variation, one FutureSeed route,
  prediction/write-key cosine at least `.999`, bounded decay/write controls,
  and token/board-dependent Q/K/P activity.

Quality against frozen P059:

- balanced at least `.955` and at least `+.01`;
- future and past each regress by at most `.005`;
- joint at least `.84` and at least `+.01`;
- total errors at most 180;
- wrong-key valid-value swaps at most 105 and at most `.60713` of errors.

Cost:

- elapsed, post-warm wall and warmed-step ratios each below `1.75x` P059;
- peak allocation below `1.15x` P059.

Every integrity, activation, quality and cost check must pass. Any miss closes
the exact configuration without rescue.

## 6. Result

R1 exact source `c36caa76` stopped before its first candidate CUDA forward.
The host FLA tree already contains a module named `fla.layers.comba`; appending
the registered external path therefore resolved the host copy and the strict
source-path assertion aborted. R2 gives the registered external layer and ops
paths explicit import priority and clears only the colliding Comba module
cache before import. No recurrence, parameter, state, data, budget or gate has
changed. No R1 quality result exists.

R2 exact source `37c4bf44` resolves and hash-verifies the registered external
files, then stops before candidate CUDA because each mixer construction
re-imported the same source into a new Python class object. The strict identity
check correctly rejected layer instances whose content was exact but whose
class object differed from the initially pinned object. R3 caches and reuses
the first verified class. No model or gate changed and R2 has no quality result.

R3 exact source `79734b26` reaches the external chunk Triton compiler, which
rejects one WY dot because the kernel's FP32 cumulative decay promotes a BF16
key product despite the source's `.to(b_k.dtype)`. The same source pattern is
present in the pinned host FLA Comba implementation. R3 therefore closes as a
non-science production compatibility failure before logits.

R4 changes only the execution compatibility boundary. It copies the exact
hash-verified external `ops/comba` package into the run's private artifact
directory and adds explicit operand-dtype casts at the two WY forward/backward
dot sites involving cumulative decay. The source checkout remains clean. The
contract asserts the original WY hash, exact two replacements, patched hash,
effective import path and unchanged chunk hash. No recurrence, parameter,
state, initialization, data, budget or registered science gate changes.

R4 exact source `e79deeab` crosses the WY forward site, then the common output
kernel rejects FP32 `q` against BF16 state. The external layer's FP32 output
correction parameter promoted the projected BF16 query. R4 therefore also has
no logits or quality result.

R5 is the final production-compatibility attempt. The same private overlay now
stages the exact layer source and explicitly casts inner-decay `p` to key dtype
and output-corrected `q` to query dtype. Together with R4's two WY casts, the
contract pins exactly four compatibility edits and both patched hashes. Any
new dtype, graph or kernel failure closes Comba on this stack; no further
compatibility patch is authorized.

Pending the R5 exact pushed-SHA CUDA contract and single formal endpoint.

## 7. Decision

Pending. A pass supports a better GDN3 live recurrence. Native FutureSeed is
retained, but this run alone cannot claim a better FS2 intervention.

## 8. Artifacts

Pending.

## 9. Lessons

The experiment is deliberately about transition coherence rather than memory
size. A compact state with a correct closed-loop edit is more plausible than a
large state whose future evidence cannot be addressed by the receiver.
