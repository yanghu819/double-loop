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

R5 exact pushed/read-back source
`753735b6a18afbc108ee3b837064e7853722ad40` is the final
production-compatibility attempt. Its clean detached worktree stages the exact
external layer and ops sources in a run-private overlay and applies exactly
four asserted dtype-boundary edits: the two R4 WY dot casts, inner-decay `p`
back to key dtype, and corrected `q` back to query dtype. It resolves every
previous import and compile failure and completes both CUDA forward and
backward on the sole registered A800.

The strict semantic gate nevertheless fails. Against the external fused
recurrent reference, the production chunk path has output relative RMS
`.239417` and terminal-state relative RMS `.258119`, both above the registered
`.05` maximum. Initial-state dependency is healthy at `.0415303`, so this is
not a dead state or zero-input artifact. It is a material disagreement between
the chunk training operator and the claimed closed-loop recurrence.

The contract exits status 1 and writes a non-science integrity `abort.json`.
The 10-epoch endpoint is not launched, so P065 has no quality score and cannot
support a better GDN3 or FS2 claim. During contract compile/autotune, 62 GPU
samples show overall mean/peak utilization `8.97%/73%`, active-sample mean
utilization `13.90%`, peak observed memory `1,112 MiB`, and mean/peak power
`82.37/200.29 W`. These are contract telemetry, not training occupancy.

## 7. Decision

Discarded at the strict CUDA semantic contract. Close this exact Comba
transfer, its compatibility overlay, correction factor, and all nearby
decay/gate/key/seed/LR/loss/batch/width/depth/duration rescues. The mechanism
may still be a useful clean-room architecture hypothesis, but this external
chunk implementation is not a valid production realization of the registered
recurrence. P059 Momentum DeltaNet plus native `[S,M]` FutureSeed remains the
strongest working result.

## 8. Artifacts

- Exact source: `753735b6a18afbc108ee3b837064e7853722ad40`.
- Clean detached worktree:
  `/huyang2/double-loop/worktrees/p-gdn3-065-r5-753735b`.
- Contract run:
  `/huyang2/double-loop/runs/p-gdn3-065-comba-futureseed-r5-20260816T082800Z-753735b`.
- External read-only source:
  `HuuYuLong/MomentumDeltaNet@c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`;
  no declared license and no copied implementation in this repository.
- `contract.log` SHA256:
  `00db633728399ffc4961641e82c3b6b36aadb521ef368ab80076b164229b0380`.
- `abort.json` SHA256:
  `873c0453e1c78115014b1f82b04f3f0abe7191d0b91f2bd7dfdad28bb794a6a3`.
- `gpu_samples.csv` SHA256:
  `8b8cae1bf2ecbd3a8ebb1b800bc19a350e7df2a05864a79335eab9dc0d581f56`.
- Source snapshot SHA256:
  `0cf329739f0ee03febdfeb72d8f7f82fbe4029d74ef2c514789869b4f16842fd`.
- Compact evidence archive SHA256:
  `5379c0fd5113f7f06d1085f2a828a290111d7813944745391b4971dbb9394b39`.

## 9. Lessons

The experiment is deliberately about transition coherence rather than memory
size. A compact state with a correct closed-loop edit remains more plausible
than a large state whose future evidence cannot be addressed by the receiver,
but operator parity is part of that claim. A recurrence is not validated when
its training chunk and recurrent reference disagree by roughly 24-26% RMS.
The next foundational transition must establish exact chunk/recurrent semantics
from the start rather than accumulate compatibility patches around an external
operator.
