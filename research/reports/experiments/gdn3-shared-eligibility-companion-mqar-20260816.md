# P-GDN3-056: Shared-Eligibility Companion GDN3

## 1. Metainfo

- Status: complete; discarded
- Date: 2026-08-16
- Branch: `codex/gdn3-shared-eligibility-20260816`
- Decision field: directional MQAR L1024 wrong-key binding regime
- Frozen control: deterministic P-REPRO-001 replay B
- Fixed model: D128/L2/H4/K32/V32 native GDN2 plus native FutureSeed
- Fixed data/training: four associations, 10,000 train and 1,000 validation
  examples, 10 epochs, batch32, seed123
- GPU: one AIStation A100-SXM4-80GB, CUDA index 0 only

## 2. Evidence And Hypothesis

The frozen native endpoint reaches balanced/future/past/joint
`.494/.454/.534/.041`. Of 2,024 errors, 1,546 are valid values bound to the
wrong key; `99.61%` of swaps choose an adjacent owner by write rank and
`99.55%` remain in the correct direction class. Deleting the competing
layer-0 write repairs `98.25%` of swaps but destroys that competitor's answer.
The payload exists, while occurrence ownership is merged during the first
live recurrent transition.

Direct erase/write-key splitting, lagged main keys, post-hoc orthogonalization,
RLS conditioning, independent banks and local attention all fail. They either
break the jointly learned query/erase/write/read frame or learn direction
correlation without one-to-one ownership. The new hypothesis is narrower:
the key should be decoupled by **role and time**, not by giving erase and write
unrelated projections. A shared token-identity read address plus a learnable
causal eligibility trace for the committed write may keep neighboring events
separable without changing the native memory plane.

## 3. Mechanism

The complete native GDN2 scan and native terminal FutureSeed remain unchanged.
Each layer adds one bounded K32xV32 companion state computed by the same pinned
official `chunk_gdn2`:

```text
a_t = normalize(P_shared * token_embedding_t)
z_t = CausalDepthwiseConv4(a_<=t)       # identity at initialization
C_t = GDN2(q=a_t, k=z_t, v=v_native, g/b/w=native; C_{t-1})
y_t = y_native + tanh(r_head) * normalize(read(C_t, a_t))
```

`P_shared` and the four-tap trace are shared across layers. The companion
terminal state is normalized and passed from layer 0 to layer 1 through its
own per-head FutureSeed gate. The read gates start at exactly zero, so full
logits, native recurrent state, native FutureSeed and all parent gradients are
identical to the frozen parent at initialization. The trace starts as the
current-token identity and can learn previous-token eligibility end to end.

The candidate adds exactly 16,912 parameters, one 4,096-value companion state
and one additional pinned-official scan per layer. It adds no token-vocabulary
rules, labels, selector, search, fixed lag, second model, quadratic attention
or Sudoku logic.

## 4. Why Existing Failures Do Not Cover It

- P-GDN3-031 split erase and write coordinates inside the main state; this
  candidate keeps coherent native erase/write and places role-time separation
  in an independent bounded plane.
- P-GDN3-051 imposed a fixed signed lag on the main key; this trace is learned,
  causal, shared and can retain current plus multiple preceding identities.
- P-GDN3-049 used a canonical companion address derived from native Q/K but no
  role-time eligibility path or companion FutureSeed.
- P-GDN3-055 added a post-main local correlation residual; this candidate
  forms a live recurrent ownership state before global readout.

## 5. Strict CUDA Contract

Exact pushed source in a clean detached worktree must prove:

1. one visible CUDA index 0 with the registered A100 UUID;
2. pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. two native plus two companion official `GatedDeltaNet2` backward paths,
   Triton ShortConv and no fallback;
4. exactly 16,912 new parameters, one 4,096-value companion state per layer,
   one shared projection and one shared four-tap causal trace;
5. zero read gates preserve full logits, arbitrary nonzero native-state
   outputs/states and all parent gradients exactly;
6. all eight read-gate heads receive finite nonzero gradient at zero;
7. opened gates produce finite nonzero shared projection, trace and layer-1
   companion-FutureSeed gradients and change logits;
8. the initialized trace is exact identity, an opened history tap is causal,
   and the receiving layer depends on nonzero companion state; and
9. data, initialization, source, GPU and artifact provenance are exact.

## 6. Fixed Science Gate

After the contract, run one candidate-only 10-epoch/batch32/seed123 arm from
the frozen P-REPRO-001 initialization. Do not repeat the control.

Activation requires all eight read gates at absolute value `>=1e-3`, both
companion outputs at relative RMS `>=.02`, finite board/token/state variation,
one active companion FutureSeed route, history mass fraction `>=.02`, event
write/read relative RMS `>=.02`, and the native FutureSeed route active.

Quality requires all of:

- balanced accuracy `>=.65` and gain over control `>=.10`;
- future and past accuracy each `>=.62`;
- joint exact `>=.15` and gain over control `>=.10`;
- total errors at least `20%` lower; and
- wrong-key valid-value swap fraction among errors at least `.10` lower.

Cost requires elapsed, post-warm wall and warmed-step time each `<2.25x`
control and peak allocation `<1.75x`. These ceilings cover one additional
linear official scan plus a depthwise four-tap trace and are frozen before the
result.

## 7. Falsifiable Prediction And Kill Rule

If role-time eligibility is the missing ownership variable, the candidate
must improve both directions, reduce adjacent-owner swaps and reach the joint
closure thresholds rather than merely changing blank/value-set accuracy. Any
integrity, activation, quality or cost miss closes this entire shared-
eligibility companion family. There is no trace-width, gate, projection,
normalization, state-size, layer, seed, LR, loss, batch, epoch or Sudoku rescue.

## 8. Results

Exact pushed/read-back source
`29688a1073b0a0dc2f3b6462bd363c8f18b9c7dc` passed the strict CUDA
contract on one A100-SXM4-80GB. The contract found four official
`ChunkGDN2FunctionBackward` paths, the exact 16,912-parameter and
4,096-state-value/layer deltas, bit-exact zero-gate logits and arbitrary
nonzero native-state parity, complete gradients, causal history dependence
and no fallback.

Run `p-gdn3-056-shared-eligibility-r1-20260815T191900Z-29688a1` completed
normally. Control versus candidate balanced/future/past/joint accuracy is
`.4940/.4540/.5340/.0410` versus `.14825/.1535/.1430/0`. Total errors rise
`2024->3407`. Wrong-key valid-value swaps fall `1546->639`, and their fraction
of errors falls `.763834->.187555`, but paired transitions show broad damage:
only `68+229=297` previously wrong queries become correct while
`1359+321=1680` previously correct queries become wrong.

The branch is active rather than dead. All eight read paths and both layer
states are live; history mass is `.09421`, event write/read relative RMS is
`.18595`, and the companion output is `1.174/2.661x` the native output RMS in
layers 0/1. The registered material-use check therefore fails because the
second layer overwhelms rather than complements the native retrieval map.
Elapsed/post-warm/warmed-step/allocation ratios are
`2.1397/2.1278/1.4587/1.2611x`, all within their frozen ceilings. Across 60
nonzero-utilization samples, GPU utilization averages `54.32%`, peaks at
`82%`, and observed memory peaks at `5,033 MiB`.

Comparison, checkpoint, contract and GPU-sample SHA256 values are
`17bd2d83...6e4d2`, `03fbc83a...199a`, `d63c267f...a87a` and
`51267a69...7258` respectively. Endpoint status is zero; launcher status two
is the expected scientific-gate rejection.

## 9. Decision

Discard P-GDN3-056. It passes integrity and cost but fails the material-use
and every absolute quality gate. Reducing wrong-owner outputs by replacing
them with unrelated errors is not binding closure. Close the complete shared
eligibility trace/projection/read-gate/state-size/normalization/training
neighborhood with no rescue and no Sudoku transfer. The next mechanism must
form a joint local event representation before the native live transition;
another additive memory plane cannot supply ownership safely.
