# P-GDN3-055: Local-Binding Hybrid GDN3

## 1. Metainfo

- Status: approved; implementation complete; awaiting strict CUDA contract
- Date: 2026-08-16
- Branch: `codex/gdn3-local-binding-hybrid-20260816`
- First decision field: directional MQAR L1024 wrong-key binding regime
- Frozen control: deterministic P-REPRO-001 replay B
- Fixed model: D128/L2/H4/K32/V32 native GDN2 plus native FutureSeed
- Fixed data/training: four associations, 10,000 train and 1,000 validation
  examples, 10 epochs, batch32, seed123
- GPU: one task-mode CUDA index 0 only

## 2. Evidence And Hypothesis

The frozen native endpoint reaches balanced/future/past/joint
`.494/.454/.534/.041`. Of 2,024 errors, 1,546 are valid values owned by
another key, and `99.61%` of those swaps select the adjacent owner by write
rank. Deleting the wrong owner's layer-0 write repairs `98.25%` of swaps but
destroys that owner's answer. Values exist; nearby pair identities merge in
the first-layer recurrent trajectory.

Direct write/erase-key decoupling, anchored and biorthogonal variants,
independent banks, exact dual keys, collision losses, sparse slots, Raven
address controllers and block-RLS conditioning all fail. They alter a
co-adapted query/erase/write/read coordinate system after the fact. The new
hypothesis is different: preserve the exact native global path and learn a
local causal representation that binds nearby token evidence before the next
GDN2 layer transports it globally. If pair identity can be formed locally and
then stored by the unchanged recurrent carrier, the candidate should preserve
the native optimization transition while reducing adjacent-owner swaps.

## 3. Mechanism

Each layer retains its complete pinned-official GDN2 scan and native terminal
FutureSeed. In parallel, the same normalized mixer input enters one standard
causal scaled-dot-product-attention call over non-overlapping 128-token blocks:

```text
L = SDPA(Q_local x, K_local x, V_local x; causal, block=128)
y = y_GDN2 + W_local_o(tanh(g_head) * L)
```

The local branch has four H4/D32 heads and bias-free D128 Q/K/V/output
projections. Its four per-head gates start at exactly zero in each layer, so
the full candidate output, native GDN2 terminal state and FutureSeed transport
are parent-identical at initialization. It adds 65,540 parameters per layer,
131,080 total, zero persistent state and no new global recurrent scan. Eight
length-128 blocks are batched into one FlashAttention SDPA call per layer.

Layer 0 can bind local key/value evidence into the residual stream; layer 1's
unchanged GDN2 can then carry that representation to distant queries. This is
not a Transformer replacement: the only global path remains linear recurrent
memory, and local attention is O(T*128*D) with fixed context.

## 4. Boundary

This is not another key transform, erase/write decoupler, cache, selector,
slot router, parallel recurrent expert, post-scan readout or Sudoku rule. It
does not carry a quadratic global attention matrix, change native recurrent
state, or reuse labels. It is trained from scratch on the validated generic
L1024 binding-error regime rather than judged by a 100-step Sudoku graft.

The only intervention is the fixed local binding path. There is no window,
gate, projection, head, depth, seed, LR, loss, batch, epoch, FutureSeed or
training-duration sweep.

## 5. Strict CUDA Contract

Exact pushed source in a clean detached worktree must prove:

1. one visible CUDA index 0 with the registered UUID;
2. pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. two exact official `GatedDeltaNet2` carriers,
   `ChunkGDN2FunctionBackward`, Triton ShortConv and no fallback;
4. exactly two FlashAttention backward paths, one fixed-block call per layer;
5. exactly 131,080 new parameters and zero persistent-state delta;
6. zero gates preserve full logits, arbitrary nonzero incoming-state outputs,
   terminal states, native parent gradients and FutureSeed transport exactly;
7. every gate head receives a finite nonzero gradient at zero, and opened
   gates give finite nonzero Q/K/V/output projection gradients;
8. the opened path changes logits, is strictly causal within a block, depends
   on earlier same-block tokens and is exactly independent across blocks; and
9. source, data, initialization and GPU provenance are exact, with no CPU
   model path, concurrent model/eval, NaN, OOM or silent fallback.

## 6. Fixed Science Gate

After the contract, run one candidate-only 10-epoch/batch32/seed123 arm from
the SHA-locked P-REPRO-001 initialization. Do not repeat the frozen control.

Activation requires all eight gates to have absolute value at least `1e-3`,
local-output relative RMS at least `.02` in both layers, finite nonzero board
and token variation, and the native FutureSeed route active.

Quality requires all of:

- balanced accuracy `>=.65` and gain over control `>=.10`;
- future and past accuracy each `>=.62`;
- joint exact `>=.15` and gain over control `>=.10`;
- total errors at least `20%` lower; and
- wrong-key valid-value swap fraction among errors at least `.10` lower.

Cost requires fit elapsed, post-warm wall and independent warmed-step time
each `<2.25x` control, with peak allocation `<1.75x`. These ceilings reflect
one fixed O(T*128*D) local branch and are frozen before observing results.

## 7. Falsifiable Decision

A complete pass admits exactly one hard-Sudoku transfer with the same native
GDN2 and FutureSeed scaffold. Any integrity, activation, quality or cost miss
closes the entire fixed-block local-binding hybrid. A miss does not authorize
window, overlap, gate, projection, head, layer, seed, optimizer, training or
Sudoku rescue.

## 8. Pending Provenance

Exact source SHA, run name, contract, score, checkpoint, source snapshot, GPU
samples and final decision will be written only after the pushed-SHA run
finishes.
