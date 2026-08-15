# P-GDN3-057: Raven/GDN Pair-Event Encoder

## 1. Metainfo

- Status: registered; implementation complete, CUDA contract pending
- Date: 2026-08-16
- Branch: `codex/gdn3-pair-event-encoder-20260816`
- Decision field: directional MQAR L1024 wrong-key binding regime
- Frozen control: deterministic P-REPRO-001 replay B
- Fixed model/data: D128/L2/H4/K32/V32, four associations, 10,000 train
  and 1,000 validation examples, 10 epochs, batch32, seed123
- GPU: one AIStation A100-SXM4-80GB, CUDA index 0 only

## 2. Evidence And Hypothesis

Native GDN2 plus FutureSeed reaches balanced/future/past/joint
`.494/.454/.534/.041`. Of 2,024 errors, 1,546 are valid values assigned to
the wrong key; almost every swap selects the adjacent write owner. Removing
that competing layer-0 write repairs `98.25%` of swaps but destroys the other
binding. The failure is superposition during the first live state formation,
not missing values or insufficient cross-layer transport.

P-GDN3-055 proves that a post-GDN local-attention residual learns direction
but not symmetric ownership. P-GDN3-056 proves that an additive eligibility
memory can reduce wrong-owner outputs only by overwhelming native retrieval.
The new hypothesis is that a key and its local payload must first become one
learned event representation, before GDN2 derives Q/K/V and commits to global
memory.

## 3. Mechanism

Each layer adds a Raven/RWKV-style causal token-shift pair encoder before the
unchanged native GDN2:

```text
x_t = rms_norm(h_t)
p_t = silu(W_current x_t) * W_previous x_{t-1}
d_t = head_rms_norm(W_output p_t) * tanh(g_head)
h_event_t = h_t + d_t
y_t, S_t = native_GDN2(h_event_t, S_{t-1})
```

`g_head` starts at exactly zero. The parent logits, recurrent state,
FutureSeed, parent gradients and one pinned-official scan per layer are
therefore exact at initialization. The rank-64 multiplicative event feature
is learned end to end and is generic: it uses no token IDs, labels, pair
locations, fixed write suppression, selector, search or Sudoku rule.

The model adds exactly 49,160 parameters, no recurrent state and no extra
scan. It is the requested organic Raven/GDN hybrid: local gated token shift
forms an event, GDN2 stores it globally, and native FutureSeed carries the
terminal global state across layers.

## 4. Why Existing Failures Do Not Cover It

- P-GDN3-051 shifted only the GDN address. P057 jointly changes the hidden
  source of Q, K, V and all gates through a multiplicative current/previous
  event feature.
- P-GDN3-055 added local context after the native GDN output. P057 forms local
  identity before the live recurrent commit.
- P-GDN3-056 added a second memory/read plane. P057 keeps one native state and
  cannot replace its readout with a side answer.
- Direct decoupled-key, whitening, banks and RLS modify mature address
  geometry. P057 leaves the native projections and recurrence untouched and
  improves the representation they receive.

## 5. Strict CUDA Contract

Exact pushed source in a clean detached worktree must prove:

1. one visible registered A100 at CUDA index 0 and pinned FLA SHA
   `9c8e42e762fce087c27b673af4922795d9edb85e`;
2. exactly two official `ChunkGDN2FunctionBackward` paths and three Triton
   short convolutions per native layer, with no fallback;
3. exactly 49,160 new parameters, zero state/scan delta, rank64 and two exact
   pair-event mixers;
4. zero gates preserve full logits, arbitrary nonzero incoming-state
   outputs/states and every parent gradient exactly;
5. all eight zero gates receive finite nonzero gradient;
6. opened gates give finite nonzero gradients to current, previous and output
   projections and materially change logits;
7. exact causality, current-token and previous-token dependence, no two-token
   leak, and exactly zero event at the first token; and
8. frozen data, initialization, source and artifact hashes remain exact.

## 6. Fixed Science Gate

After contract success, run one candidate-only 10-epoch/batch32/seed123 arm
from the frozen initialization. Do not repeat the control.

Activation requires all eight gates at absolute value `>=1e-3`; both layer
event deltas must have relative RMS in `[.02,.35]`; current/previous features
and board/token variation must be finite and nonzero; native FutureSeed must
remain active.

Quality requires all of balanced accuracy `>=.65` and control gain `>=.10`,
future and past accuracy each `>=.62`, joint exact `>=.15` and gain `>=.10`,
total errors at least `20%` lower, and wrong-key swap fraction at least `.10`
lower.

Cost requires elapsed, post-warm wall and independently warmed step each
`<1.50x` control and peak allocation `<1.25x`. These ceilings are frozen for
three rank-64 linear maps and one elementwise product per layer.

## 7. Falsifiable Prediction And Kill Rule

If local joint event formation is the missing ownership step, both query
directions must improve together, native-correct retention must rise, and
adjacent-owner swaps must fall without broad retrieval damage. Any integrity,
activation, quality or cost miss closes the complete one-shift multiplicative
pair-encoder family, including rank, gate, normalization, projection, layer,
seed, LR, loss, batch, epoch and Sudoku rescue.

## 8. Results

Pending.

## 9. Decision

Pending.
