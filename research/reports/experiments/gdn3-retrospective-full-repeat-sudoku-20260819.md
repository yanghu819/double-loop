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

Overall decision: **discarded: both quality routes and the allocation gate
failed**. The exact full-repeat teacher is closed on this Sudoku parent, and no
compiled student or sparse replay derived from it is authorized.

The exact pushed source was
`0b531f69b250442f61ba0852a35c466d508daa88`. The strict R3 CUDA contract passed
on A100 80GB index0, UUID
`GPU-573c7ed1-1c51-8334-299b-edf2ff3440e6`, with pinned FLA source
`9c8e42e762fce087c27b673af4922795d9edb85e`. It proved ten official
`GatedDeltaNet2` layers and ten `ChunkGDN2FunctionBackward` paths, exact zero
parameter/state delta, nine receiving length162 paths, bit-identical Q/K/V/g/b/w
halves, nonzero incoming FutureSeed, and finite nonzero Q/K/V/decay/erase/write
gradients in all receiving layers. The contract's output relative RMS was
`.784538`, with finite board variation and bounded terminal states.

The exact-resume step12001 probe exited status0 and verified optimizer/RNG/data
order and production activation; its eight-board score was not used as quality
evidence. The formal step12000->12100 continuation also exited status0 with
complete metrics, checkpoint, config, source snapshot and same-board case banks.
There was no NaN, OOM, fallback, source drift, data drift, or GPU drift.

### Official full-diversity result

Each cell is `full-board exact / blank accuracy / mean wrong blank cells` on
the same 512 boards. Mean wrong cells use the fixed official bank's mean blank
counts (`54.6152`, `56.8848`, and `63.9902`).

| Range | Arm | loop1 | loop2 | loop3 | loop4 | loop5 |
|---|---|---|---|---|---|---|
| 51-55 | control | `.0000/.6185/20.84` | `.0469/.7296/14.77` | `.3301/.7878/11.59` | `.4199/.7979/11.04` | `.4316/.8002/10.91` |
| 51-55 | candidate | `.0000/.6084/21.39` | `.0293/.7151/15.56` | `.2949/.7671/12.72` | `.3691/.7790/12.07` | `.3906/.7809/11.97` |
| 56-60 | control | `.0000/.5290/26.79` | `.0137/.6052/22.46` | `.1289/.6389/20.54` | `.1855/.6485/20.00` | `.1973/.6500/19.91` |
| 56-60 | candidate | `.0000/.5230/27.13` | `.0156/.5895/23.35` | `.1016/.6265/21.24` | `.1484/.6350/20.77` | `.1582/.6366/20.67` |
| 61-64 | control | `.0000/.5736/27.29` | `.0000/.7327/17.11` | `.0820/.8280/11.00` | `.2500/.8584/9.06` | `.2891/.8645/8.67` |
| 61-64 | candidate | `.0000/.5585/28.25` | `.0000/.7066/18.78` | `.0586/.7981/12.92` | `.1699/.8282/10.99` | `.1992/.8339/10.63` |

At loop5 the exact deltas are `-.041016/-.039063/-.089844`, and blank deltas
are `-.019347/-.013391/-.030614`. Hard51-64 macro exact falls from `.305990`
to `.249349`, delta `-.056641`; it misses the registered `.320990` primary
floor. Loop3-to-loop5 wrong-cell correction is `.752/.572/2.291` for the
candidate versus `.676/.633/2.336` for control. The candidate improves late
correction only in 51-55, remains worse at loop5 in every range, and cannot
satisfy the alternate route because 61-64 exact regresses sharply.

### Mixed result, optimization and activation

| Arm | loop1 exact/blank | loop2 | loop3 | loop4 | loop5 |
|---|---|---|---|---|---|
| control | `.0234/.5758` | `.0293/.6713` | `.2480/.7186` | `.3125/.7311` | `.3262/.7332` |
| candidate | `.0234/.5712` | `.0293/.6646` | `.2285/.7101` | `.3086/.7199` | `.3223/.7212` |

Mixed loop5 exact falls `-.003906`, far below the alternate `+.02` floor, and
mixed blank accuracy falls `-.012063`. Training CE is
`.612266` for control and `.618129` for the candidate. The candidate recovered
from CE `2.0266` at step12001 to near-control CE by step12100, so the endpoint
is not an unadapted one-step probe; quality still remained lower.

Formal activation is material and stable: exactly nine receiving paths,
second-minus-first output relative RMS `.530087` during training and `.533215`
at loop5 evaluation, board standard deviation `.127156/.140340`, first/second
output RMS `.622096/1.007354`, and loop5 terminal RMS/board standard deviation
`4.297377/1.081063`. The negative quality result is therefore not a dead path
or silent fallback.

### Cost and GPU evidence

The full step12100 checkpoint/evaluation elapsed time is
`698.848->725.520` seconds (`+3.817%`), passing the preregistered `<2.20x`
teacher bound. Training-only `train_sec` is `783.176->760.693` seconds; both
scopes pass, and the discrepancy reflects run/evaluation timing rather than a
quality claim. Peak allocated memory rises `8095.40->10191.89 MiB`, ratio
`1.258973`, strictly failing the `<1.25x` gate. Reserved memory rises
`8588->10910 MiB` (`+27.038%`). Parameter count remains exactly `5,461,688`.

Five-second samples over the formal stage show mean utilization `25.53%`,
busy-sample mean `28.68%`, all-sample p90 `45%`, busy-sample p90 `48%`, and
maximum `84%`. Mean/peak sampled device memory is `11930.7/14097 MiB`. The
small D192, length81 model is launch- and evaluation-heavy; the low average is
not evidence of a second concurrent workload.

### Same-board evidence

Candidate and control export the same 256-board visualization banks for each
range. Their immutable data hashes are:

- 51-55: `8af125d2f365ef7b62b14724fe5cb885df2de855130a25c6a5adbc6ffb6507ac`;
- 56-60: `a362d2adddaf9297865b76aa064670ab87ad8a1a009f89a41c06d1612031d054`;
- 61-64: `bed22b6b6916569f62f521a5493150237b7c270e3b9ef95c1449cb54f5629cb3`.

Each run contains `index.html`, selected `cases.json`, and complete
`all_cases.json` under `output/case_bank/official_b*/`. This is genuine
same-board loop1-5 evidence, not an aggregate probe or hardest-board claim.

## 9. Provenance

- **Integrity:** pass after the unchanged R3 checker fix. R2 failed only because
  the checker treated official FLA's sequential `f_proj` as a bare linear
  layer; it exited before a science score. No mechanism, setting, or gate was
  changed.
- **Activation/stability:** pass.
- **Primary quality:** fail (`-.056641` hard macro exact and all blank ranges
  regress by more than `.01`).
- **Alternate quality:** fail (`-.003906` mixed exact, 61-64 regression, and
  weaker 56-60/61-64 late correction).
- **Elapsed cost:** pass (`+3.817% <120%`).
- **Allocation cost:** fail (`1.258973x`, not `<1.25x`).
- **Final:** discard exact full repeat. Do not launch its K=8 compiled student
  or nearby router/cache/repeat rescue on this parent.

Formal artifact hashes:

- candidate metrics/checkpoint/config/source snapshot:
  `9f2a89b0...9e38`, `0c3fba48...2a8b`, `a9ba3460...096`,
  `a55d8921...c1e`;
- R3 contract/sequence log/GPU CSV/manifest/status:
  `4425ccdc...eff`, `d988a2a6...95e5`, `73489d3f...0a67`,
  `d29ad2f3...ecae`, `9a271f2a...86aa`.

The full hashes and structured gate result are stored in
`gdn3-retrospective-full-repeat-sudoku-20260819-comparison.json`, SHA256
`d229fc35b2abcc8b8967b3bb70cabb2dd9fe0ac55fcd09b1ba170081aee780e7`.
