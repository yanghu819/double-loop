# P-FS2-013: Loop-Secant FutureSeed

## 1. Metainfo

- Status: complete; discarded at the matched quality and elapsed-cost gates
- Date: 2026-08-15
- Branch: `codex/fs2-loop-secant-20260815`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: one AIStation A100 80GB, CUDA index 0 only
- GPU UUID: `GPU-d2877fe4-641c-fe64-2a74-8abca47c292f`
- Seed: 52 only
- Parent: D256/L12/H8/K32/V32 position-QK GDN3 plus native terminal
  FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen control: `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Evidence Boundary

The current evidence separates two failures. Directional MQAR localizes the
dominant GDN2 failure to value evidence being committed to the adjacent
same-direction owner; exact dual-key reconstruction then shows that changing a
trained address basis after the fact destroys the jointly learned query,
erase, write and read closure. Sudoku shows a different endpoint symptom:
native FutureSeed often makes genuine corrections in loops 2-3 but then
plateaus or oscillates rather than closing the board.

Prior loop mechanisms do not test a directional derivative of the transported
recurrent state. `loop_residual` interpolated the previous and current seed;
learned update gates changed only interpolation rate; logit feedback, gated
scratch and extra loops did not create late exact closure. P-FS3-001 changed
within-edge terminal content using an innovation residual, not the direction
of change across repeated reasoning passes.

## 3. Mechanism

For each adjacent producer-to-receiver edge `e` and head `h`, retain the
producer terminal state from the preceding application of the same reasoning
stream. Let the current and preceding producer terminals be `T_r` and
`T_{r-1}`. Form the secant

`D_r = T_r - T_{r-1}`.

Bound its RMS to the current terminal RMS and extrapolate from the current
state:

`S_seed = T_r + 0.5*tanh(a_eh)*bounded(D_r)`.

The first pass has no history and remains native. Parameters `a_eh` are zero
initialized, so every pass is exactly the parent function at launch. On the
high stream, consecutive applications are macro loops; on the low stream they
also include consecutive L-cycle passes. This deliberately tests whether the
direction in which the recurrent state is already moving is useful future
evidence, without adding tokens, scans, caches, new recurrent state, Sudoku
logic or a reverse pass.

For D256/L12/H8, the exact parameter delta is `11*8 = 88`; persistent model
state, physical GDN2 state, scan count and official FLA kernels are unchanged.

## 4. Falsifiable Prediction

If the remaining Sudoku failure is loop convergence rather than missing
address capacity, all 88 coefficients should receive gradient immediately and
the learned secant should show nonzero board variation. A 100-step exact
continuation should improve hard-board exact closure or mixed exactness while
preserving the hardest range and strengthening same-board loop3-to-loop5
correction. If the path activates but exact remains flat or correction weakens,
cross-pass state direction is not sufficient and this family is closed. If the
strict identity, official-kernel or RMS bound fails, the implementation claim
is false before science scoring.

## 5. Strict CUDA Contract

Exact pushed source in a clean detached worktree must prove all of:

1. only CUDA index 0 and UUID
   `GPU-d2877fe4-641c-fe64-2a74-8abca47c292f` are visible;
2. pinned FLA SHA is
   `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. all 12 layers are official `GatedDeltaNet2`, with
   `ChunkGDN2FunctionBackward` in every layer graph and no fallback;
4. zero initialization preserves full output and all 12 terminal states
   bit-exactly across five varying passes;
5. arbitrary finite nonzero prior-pass producer memories also preserve output
   and all states bit-exactly at zero initialization;
6. returned producer memory equals the live producer terminal state exactly;
7. parameter delta is exactly 88 and all 88 coefficients receive finite,
   nonzero direct gradients;
8. an opened path changes output, has nonzero state delta and board variation,
   remains finite, and satisfies coefficient and residual RMS bounds `<=0.5`;
9. no CPU model path, concurrent GPU process, NaN, OOM, source/data drift or
   silent fallback occurs.

## 6. Step3001 Production Gate

The one-step exact-resume probe is migration and production-fit evidence only,
not a quality measurement. It must:

- accept only the explicit `fixed -> loop_secant` semantic upgrade;
- add exactly `reasoner.future_seed_secant_raw` to optimizer groups while
  preserving optimizer/RNG/data order;
- complete status 0 with source/config/checkpoint/metric provenance;
- report finite nonzero secant scale, residual relative RMS and board variation
  after one update;
- retain the target GPU, official FLA graph and clean source contract.

## 7. Matched Science Gate

Only after Sections 5-6 pass, run one candidate-only step3000-to-step3100
continuation. Do not repeat the frozen control.

- Primary quality route: hard51-64 macro loop5 exact improves by `>=0.02`
  over control and each official hard range blank accuracy regresses by no more
  than `0.01`.
- Alternate quality route: mixed loop5 exact improves by `>=0.03`, official
  61-64 does not regress, and same-board loop3-to-loop5 wrong-cell correction
  is stronger.
- Activation: mean absolute secant coefficient `>=1e-4`; delta/residual RMS and
  board variation finite and nonzero.
- Cost: independent warmed elapsed overhead `<10%` and peak allocation
  overhead `<10%` versus the frozen control protocol.

Any integrity, activation, quality or cost miss discards P-FS2-013. There is no
coefficient cap, initialization, edge subset, seed, LR, loss, batch, width,
depth or duration rescue.

## 8. Required Endpoint Evidence

Report official 51-55/56-60/61-64 and mixed loop1-5 exact, blank accuracy and
wrong-cell counts; train CE; secant coefficient, delta and residual metrics;
same-board loop trace; independent warmed throughput; peak allocated/reserved
memory; timing stability; source/config/metric/checkpoint hashes.

## 9. Decision

Discarded. Contract R1-R4 localized harness and then implementation defects
before any science run. R5 fixes the common-tail overwrite that had disabled
FutureSeed, then passes exact parent output/state identity, nonzero incoming
state identity, all 88 direct gradients, official FLA provenance, bounded
geometry and the target GPU contract. Its log SHA256 is
`de3d42d11ca13e39f110f982e7283f0251750d8020792b04b50c0819880f36f7`.

The exact-resume step3001 production probe completes status 0. Its metrics and
checkpoint SHA256 are
`aaa5e8e0ef1caab0f5a5c202540e2e1940c9fa35ab6f552a5e4daa8eaad4c4b0`
and `d205b33c0d47c031f9277144c70c868b64712db7a6a6ca8a7229d34c5bf11935`.
The formal candidate
`p-fs2-013-loop-secant-s3100-20260815T145751Z-6798f03` then completes
status 0 from exact step3000 resume. Train CE is `.860895` versus frozen
control `.858617`; parameter count is `11,485,848`, exactly 88 above control.

The endpoint rejects both quality routes. Hard51-64 macro loop5 exact is
unchanged at `.000651`; mixed loop5 exact is unchanged at `.025391`.
Candidate mixed exact over loops1-5 is
`.015625/.023438/.025391/.025391/.025391`, while blank accuracy is
`.510961/.537464/.544002/.544142/.543862`. The frozen control is
`.017578/.023438/.025391/.025391/.025391` exact and
`.515087/.536904/.544841/.545540/.545610` blank.

Official loop5 candidate versus control exact/blank is:

- 51-55: `.001953/.574840` versus `.001953/.573766`;
- 56-60: `0/.502248` versus `0/.503861`;
- 61-64: `0/.587741` versus `0/.591923`.

The secant is genuinely active: mean absolute/signed coefficient is
`.0069388/.0012313`. At loop2, delta/residual relative RMS and residual board
standard deviation are `.408061/.002687/.002242`; by loop5 they contract to
`.005297/.00004161/.00008337`. Thus this is not a dead-path result. It says
that extrapolating the already-computed state direction produces a rapidly
vanishing perturbation and does not restore missing ownership information.

Same-board traces reinforce that conclusion. Candidate versus control mean
wrong cells over loops1-5 are `25.81/24.39/24.13/24.04/24.11` versus
`25.74/24.35/23.92/23.86/23.93` for 51-55,
`29.71/28.29/27.96/27.92/27.87` versus
`29.70/28.20/28.08/28.04/28.02` for 56-60, and
`31.97/27.37/26.72/26.55/26.57` versus
`31.86/27.20/26.61/26.51/26.43` for 61-64. Loop3-to-loop5 correction is
`.0234/.0898/.1484` versus control `-.0156/.0664/.1836`; the hardest range
weakens and loop5 has 90 better, 62 equal and 104 worse boards.

Fresh-process 100-step throughput is `13.093` effective boards/s versus
`15.497`; elapsed overhead is `18.36%`, failing the registered `<10%` gate.
Peak allocated/reserved memory is `13,800/14,998 MiB` versus
`13,186/14,288 MiB`, passing at `+4.66/+4.97%`. The 2-second GPU sampler has
508 active samples with mean/median/p90/peak utilization
`28.12/26/37/90%`, mean memory `17,333 MiB` and observed peak `18,259 MiB`;
it includes compile, train, checkpoint and evaluation phases.

Formal metrics/checkpoint/config/run-log/source-snapshot/GPU-sample SHA256 are
`dd302347eb023ce5abc884ec982a6d838cbc661166545412b9d4dcf280027a63`,
`35f0de15a29bc7144cc7d1ccb1c8df5ab0f06c4164f69c5a42922202cc4c3dad`,
`2cc671cf882a0e78b0b4c41cca3edfe4382c7b32aa20c89da0e556371e299828`,
`c133ed2d34222c66f19dd1eea5145bd52456ff85dc5d3d34fca2e2df9f4ca582`,
`bcc14a034c3b1e8dc91228d590aa0ca2a4d3de38b6cb6676117b476eddba5208`
and `9b4c605e1738b74dcc15c7633f487f9bfd559de01006ff526227374c845f81cd`.
Close secant cap, initialization, edge subset, seed, LR, loss, batch, width,
depth and duration rescue. The next FutureSeed mechanism must carry new
ownership evidence or change credit assignment; it cannot merely accelerate
the same lossy state trajectory.
