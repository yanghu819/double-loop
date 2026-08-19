# P-GDN3-074 / P-FS2-020 Retrospective Full-Repeat Teacher on Sudoku

## 1. Research Question

Does a tied second recurrent pass over an already observed Sudoku context
provide a real quality ceiling above native FutureSeed, before spending compute
on a router or a compiled-seed student?

## 2. Evidence And Hypothesis

Native FutureSeed is causal and strong, but it transports only one producer
terminal state. P072 then showed that receiving layers lose useful inherited
content during their live scan: bounded boundary recommit raised hard51-64
macro loop5 exact from `.305990` to `.326172`, although its allocation cost
missed the production gate.

The attached Retrospective State Compiler proposal makes a stricter ordering
claim: full repeat must first establish a teacher ceiling; only then should a
sparse replay or state compiler be trained to approximate it. The falsifiable
hypothesis is that the same receiver, after one complete pass, can use its
whole-board state to reinterpret the same token features on pass two and close
more boards than the frozen native-FutureSeed control.

## 3. Fixed Intervention

For receiving layers `l=1..9`, compute native content-driven projections once:

```text
Q,K,V,g,b,w = ProjectAndShortConv(x[1:81])
(O,S2) = OfficialChunkGDN2(
    [Q|Q], [K|K], [V|V], [g|g], [b|b], [w|w],
    initial_state=NativeFutureSeed
)
output = O[82:162]
```

The two halves are bit-identical before the recurrent scan. ShortConv runs
once in original order, so its cache does not introduce a boundary artifact.
The repeated stream is processed by one pinned-official `chunk_gdn2` call,
not two Python scans. Layer0 has no incoming FutureSeed and remains the parent
path. The mechanism adds zero parameters, zero persistent state, no selector,
no retrieval bank, no task rule, and no changed loss.

This is not the existing five outer loops: outer loops update hidden streams
and feed predictions back between passes. P074 repeats the exact same projected
context inside one receiving recurrent layer. It is also not P072: no inherited
component is projected or manually recommitted at token64.

## 4. Fixed Matched Protocol

- Parent checkpoint:
  `/huyang2/double-loop/models/gdn2-futureseed-clean-scale-s12000-final-20260802T163724Z-5889462/checkpoints/train_state_step012000.pt`.
- Parent SHA256:
  `66b805cf163a6b0eaae1ec6ed74f9f7a8b3cc6e21919b58268b8150ae5ae3b37`.
- Parent source: `5889462cb9234ee8632a2dcb3c0690dee2f82e0d`.
- Model: pinned-official FLA GDN2 D192/L10/H6/K32/V32, native terminal
  FutureSeed, loop5 equal CE, BF16, effective batch128, seed52.
- Data: unchanged official/full-diversity hard 9x9 Sudoku 51-64 continuation.
- Candidate: exact resume of optimizer/RNG/data order for exactly 100 steps,
  step12000->12100.
- Frozen control: `p-gdn3-072-control-s12100-20260818T062611Z-c58ff80`;
  no repeated control.

Frozen loop5 reference: hard51-64 macro exact `.3059895833`, mixed exact
`.326171875`; official 51-55/56-60/61-64 exact
`.431640625/.197265625/.2890625` and blank
`.800200284/.649991393/.864542305`.

## 5. Contract And Activation Gates

Before training, one strict CUDA contract must prove:

- exactly one visible requested GPU and pinned FLA source
  `9c8e42e762fce087c27b673af4922795d9edb85e`;
- ten official `GatedDeltaNet2` layers and official
  `ChunkGDN2FunctionBackward` provenance;
- parameter delta exactly zero and exact parent state-dict schema;
- exactly nine length162 repeat calls, each with bit-identical first/second
  Q/K/V/g/b/w halves and nonzero incoming FutureSeed;
- one official chunk backward node per receiving layer;
- finite nonzero gradients through Q/K/V/decay/erase/write in all receiving
  layers, no fallback, and terminal-state RMS at most `8x` control in the
  synthetic contract.

The exact-resume step12001 probe must complete with optimizer/RNG/data-order
integrity. Formal activation requires exactly nine paths, second-minus-first
output relative RMS `>=1e-4`, finite nonzero board variation, and loop5 terminal
RMS no more than `4x` the frozen control.

## 6. Quality, Cost, And Kill Gates

Primary quality:

- hard51-64 macro loop5 exact `>=.3209895833` (control `+.015`);
- every official range blank accuracy regresses by at most `.01`.

Alternate quality:

- mixed loop5 exact `>=.346171875` (control `+.02`);
- official 61-64 loop5 exact does not regress below `.2890625`;
- same-board loop3-to-loop5 wrong-cell correction is stronger.

Because this arm is a teacher ceiling rather than a serving architecture, the
preregistered cost limits are elapsed `<2.20x` and peak allocation `<1.25x`
control. Any integrity, activation, stability, quality, or cost miss closes
this exact mechanism. Do not rescue repeat count/order, gate, temperature,
router, seed, LR, loss, batch, width, depth, or duration.

## 7. Next Decision

A quality pass authorizes one separately registered fixed K=8 receiver-native
compiled student, trained to retain the teacher gain with lower cost. A quality
miss stops router/cache/compiler work on this Sudoku parent. It does not claim
that all from-scratch retrospective architectures or language/agent critics are
invalid; they are outside this fixed Sudoku experiment.

## 8. Result

Pending.

## 9. Provenance

Pending pushed source, contract, probe, formal metrics, checkpoint, config,
logs, visualization, hashes, and final decision.
