# P-GDN3-017: Raven-Routed GDN

## 1. Metainfo

- Status: discarded after complete strict contract, exact step3001 probe and
  matched step3000-to3100 endpoint
- Date: 2026-08-10
- Branch: `codex/gdn3-raven-routed-update-20260810`
- Formal source SHA: `e884b7df948334ba6ca326717414cb3045b0d5bb`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: one AIStation task-mode A100-SXM4-40GB only, CUDA index0, UUID
  `GPU-bfb964ca-068f-8fd1-8f27-463dca141125`
- Formal run:
  `p-gdn3-017-raven-routed-s3100-20260810T065231Z-e884b7d`
- Required GPU identity: bind `EXPECTED_GPU_UUID` to the first admitted task's
  CUDA index0 UUID before any CUDA import, then require that exact UUID and one
  compute app throughout. The platform disables Stop while losing admission
  requests are Pending; if either becomes admitted, stop it before any model
  process starts.
- Seed: 52 only
- Parent: D256/L12/H8/K32/V32 position-QK GDN3 plus native terminal
  FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Evidence Boundary

The original full Raven replacement is already rejected at this scale. It was
slower than GDN2 and failed the hard Sudoku carrier. That result tests Raven as
the complete recurrent core; it does not test whether Raven's useful idea,
content-dependent memory allocation, can control a stronger GDN transition.

P-GDN3-005 through P-GDN3-016 also bound the nearby alternatives. Scalar gate
coherence, pre-scan state feedback, a parallel residual expert, terminal
consolidation, chunk-boundary transport, extra banks, interleaved writes and
Bi-Axis lifetime changes all activated without reliable exact closure. None of
them gives the existing live K-row memory a token-dependent allocation policy.

P017 therefore tests one new mechanism boundary: Raven supplies an allocation
control plane, GDN2 remains the only recurrent transition and data plane, and
native FutureSeed remains the only cross-layer state transport. It is not a
Raven replacement, another state bank, a second scan, a readout residual, or a
Sudoku-specific policy.

## 3. Mechanism

Keep the parent D256/L12/H8/K32/V32 position-QK model. In each head, partition
the existing K32 row axis into eight contiguous slots of four rows. Add one
bias-free, zero-initialized projection per layer:

`W_route: R^256 -> R^(8 heads x 8 slots)`.

For token hidden state `x_t`, compute the conserved soft allocation

`a_t = 8 * softmax(W_route x_t)`, so `mean_slot(a_t)=1`.

Repeat each slot allocation over its four K rows to obtain `rho_t`. Before the
single unchanged official GDN2 call, route its key and row decay as

`k'_t = sqrt(rho_t) * k_t`,

`g'_t = rho_t * g_t`.

Q, V, erase gate, write gate, output gate, token order and the official
`chunk_gdn2` implementation are unchanged. The in-kernel Q/K normalization
makes `k'` a content-dependent allocation of address direction across whole
row groups. Since `g<=0` and `rho>0`, decay remains nonexpansive. In the limit
`rho=0`, a slot has zero key/write contribution and zero decay, so its current
state rows are protected; other slots remain available for new writes.

At zero initialization, `a=rho=1` exactly, so output and terminal state reduce
bit-exactly to the parent, including nonzero incoming FutureSeed. Unlike hard
top-k or Gumbel routing, this reduction retains a direct first-order gradient
at the parent. P017 fixes eight slots and has no temperature or top-k knob.

The candidate adds exactly 196,608 parameters, no recurrent state values, no
tokens, no scan, no second core and no task logic.

## 4. Falsifiable Prediction

If hard-board failure is partly caused by every token updating and forgetting
the same undifferentiated K rows, the router should learn nonuniform,
board-dependent and token-dependent allocations. This should preserve some
state rows while assigning others to current writes, converting late-loop
wrong-cell reduction into more exact board closures within the matched 100
steps.

The hypothesis is false at this parent if the route is active and diverse but
hard macro and mixed exact do not improve, or if allocation collapses, state
geometry becomes unstable, late-loop correction weakens, or systems cost
exceeds the registered bound. A zero or symmetry-cancelled first-order gradient
is an implementation-level falsification before training.

## 5. Migration And CUDA Contract

The exact pushed source in a clean detached worktree must pass all of:

1. CUDA index0 and the admission-bound A100 40GB UUID, with no concurrent compute
   app;
2. pinned FLA source SHA
   `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. 12 exact official `GatedDeltaNet2` layers and one
   `ChunkGDN2FunctionBackward` per layer;
4. exact parameter delta 196,608 and zero state/token/scan/core delta;
5. bit-exact zero-init full output and all 12 terminal states, including
   synthetic nonzero incoming states;
6. finite nonzero direct router gradients in all 12 layers at zero init;
7. exact conserved allocation mean, nonpositive routed decay, and exact route
   formula agreement;
8. K-slot permutation equivariance and a synthetic zero-allocation slot whose
   state rows remain unchanged while active rows update;
9. opened routing changes output and every layer terminal state, remains
   finite, and keeps terminal RMS below four times the parent;
10. no backend dispatch, fallback, CPU model smoke, GPU2 use or concurrent GPU
    model/eval.

R1 exited before model construction because the launch environment pointed
`FLA_SOURCE_ROOT` at a source copy nested under an older repository, causing
the provenance check to resolve the wrong outer Git SHA. The non-science abort
is preserved with SHA256
`d4427e9b1571434c519a41ea99d082ef41dbe1ef2b890facf1f8fe305cdd765f`.
R2 removed that erroneous override and used the immutable FLA SHA marker. It
completed with status0 on the exact formal source:

- exact zero-init full output and all terminal states, including nonzero
  incoming states;
- exact parameter/state deltas `+196608/+0`;
- exactly one `ChunkGDN2FunctionBackward` in each of 12 official layers;
- minimum all-layer route gradient `0.0003039563`;
- allocation-budget error `1.192e-7`, slot-equivariance error `0`, protected
  zero-slot state error `0`;
- opened output change `0.111328`, with finite terminal RMS
  `0.065358415` versus parent `0.065358348`;
- R2 contract log SHA256
  `02809c22b1b8a693f807148151364de3a8b2079c4288f2b1e641109203c3e24d`.

Any miss closes P017. It does not authorize a slot count, grouping, top-k,
temperature, route scale, initialization, seed, LR, loss, batch, width, depth
or continuation rescue.

## 6. Step3001 Production Probe

After the contract, exact-resume the registered parent for one step with the
same optimizer, RNG, data order, BF16, effective batch128, seed52 and loop5
equal CE. The probe must write complete metrics, checkpoint, config and source
hashes and show:

- all 12 route paths enabled;
- allocation absolute deviation, slot variation and K/g relative change each
  finite and at least `1e-4`;
- board and token variation finite and at least `1e-6`;
- normalized allocation entropy in `[0.35,1]`, minimum allocation positive and
  maximum below `7.5`;
- finite terminal state geometry at every loop, with aggregate terminal RMS at
  most four times the parent;
- unchanged official-kernel provenance and no NaN/OOM/fallback.

Probe score is migration and production-fit evidence only. It cannot pass the
science gate.

The exact step3001 probe completed status0 and passed every registered bound.
All 12 route projections were active. At loop5, allocation deviation,
slot/board/token variation was
`0.080237/0.030918/0.001934/0.003546`; normalized entropy and min/max allocation
were `0.997790` and `0.706491/1.368140`. Routed K/g relative change was
`0.049376/0.097463`, router-weight RMS was `0.00143987`, and terminal RMS was
`6.589435` versus parent `6.697617`. Peak allocated/reserved memory was
`15718.6/16476.0 MiB`. Metrics/checkpoint SHA256 were
`4d1ea06f00e2a443eebcb63d35cfe45625a45fe29b8184eabfdf8561a45091bb` and
`0ddb152c7097e6c8be34d3f3a48bd30561d8fa0707d7eb4f43a06a57f7fede6b`.

## 7. Science And Cost Gates

Run one candidate-only exact continuation from step3000 to step3100. Do not
repeat the frozen control. At the endpoint, activation and stability must keep
the step3001 bounds. Quality passes by exactly one route:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with each official
   hard range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

The frozen control ran on A100 80GB, while P017 uses the first admitted A100
40GB task. Its timing and memory are therefore not treated as a same-hardware
overhead comparison. Before any CUDA result, the absolute A100 40GB
production-fit gate is fixed at independently warmed throughput at least
`12.0` effective boards/s, peak allocated memory below `20 GiB`, peak reserved
memory below `24 GiB`, and stable-step timing coefficient of variation below
`10%`. Quality remains compared against the frozen control because predictions
and data are hardware invariant. Any activation, stability, quality, timing,
memory or integrity miss discards P017 without rescue.

The formal continuation completed status0 with exact optimizer/RNG/data-order
resume. Endpoint activation is strong but no longer satisfies the probe's
non-collapse bound. Across loops1-5, allocation deviation is
`0.8267/0.7798/0.7840/0.7861/0.7867`, normalized entropy is
`0.7429/0.7694/0.7668/0.7655/0.7652`, and maximum allocation is
`7.984962/7.983979/7.983813/7.983770/7.983757`. Loop5 minimum allocation is
`0.000145`, routed K/g relative change is `0.508998/1.127739`, router-weight
RMS is `0.0156067`, and terminal RMS is finite at `7.366014`. The fixed
maximum-allocation `<7.5` stability gate therefore fails even though all 12
routers remain enabled and board/token variation is nonzero.

The A100 40GB absolute throughput and memory checks pass: 100 optimizer steps
at effective batch128 take `1014.070 s`, or `12.622` effective boards/s; peak
allocated/reserved memory is `15720.7/16898.0 MiB`. A separate timing-CV run is
not executed because the binding stability gate and both quality routes have
already failed. This avoids spending another GPU continuation on a result that
cannot pass the conjunction of registered gates.

## 8. Required Readout

All official rows contain 512 boards. Values below are loops1-5.

| Slice | Control exact | Candidate exact | Control blank | Candidate blank |
|---|---|---|---|---|
| mixed | .017578/.023438/.025391/.025391/.025391 | .021484/.023438/.023438/.023438/.023438 | .515087/.536904/.544841/.545540/.545610 | .510961/.536100/.541135/.542953/.543617 |
| 51-55 | 0/0/.001953/.001953/.001953 | 0/0/.001953/.001953/.001953 | .532345/.566677/.573945/.573623/.573766 | .534422/.565747/.573300/.573623/.573587 |
| 56-60 | 0/0/0/0/0 | 0/0/0/0/0 | .473182/.498061/.501699/.503140/.503861 | .472187/.498301/.502659/.503243/.502351 |
| 61-64 | 0/0/0/0/0 | 0/0/0/0/0 | .504319/.578523/.589207/.591954/.591923 | .500382/.569824/.581789/.583407/.583468 |

Hard51-64 macro loop5 exact is unchanged at `0.000651`; mixed exact regresses
by `-0.001953`. Loop5 blank deltas for 51-55/56-60/61-64 are
`-0.000179/-0.001510/-0.008455`. Train CE changes
`0.858617->0.867753`.

The matched 256-board banks have identical case IDs, labels and data hashes.

| Range | Control mean wrong cells loops1-5 | Candidate mean wrong cells loops1-5 | Control L3-L5 | Candidate L3-L5 | Candidate L5 better/equal/worse |
|---|---|---|---:|---:|---:|
| 51-55 | 25.742/24.348/23.918/23.855/23.934 | 25.918/24.227/23.859/23.742/23.703 | -0.016 | 0.156 | 108/48/100 |
| 56-60 | 29.695/28.195/28.082/28.043/28.016 | 29.836/28.355/28.063/28.074/28.070 | 0.066 | -0.008 | 102/52/102 |
| 61-64 | 31.855/27.203/26.613/26.508/26.430 | 32.004/27.613/26.887/26.754/26.723 | 0.184 | 0.164 | 87/45/124 |

The candidate improves late correction only on 51-55. It weakens on 56-60
and 61-64, so the alternate route fails independently of its mixed-exact
regression.

Formal artifact SHA256:

- metrics: `73f6304079110881544c95d7b83aae67cbb04edaff898fbb3a8848158c251b9e`;
- checkpoint: `1712f5e6318b31c137fd517a3aaa9bce319d6386689e1907686360da945ac37a`;
- config: `4aaaec99b4db1045d2c1b7efb033a2228922ab963f9ffc04aa442a8989977db7`;
- source snapshot: `1d224042557432ae3cfa55d2dab37000092763464f5bc4d9f10baef4c390fbec`;
- run log: `74fec0dbc66d119027bb5111f4af99d54e2aebb0f8706555b275f4dde026579c`;
- launch log: `eb9653e1e16dbeee4f8704ace628db9f50f5ade811748d2b3cc18065ac603666`.

Machine-readable comparison, `abort.json`, and same-board loop1-5 HTML are in
`/huyang2/double-loop/runs/p-gdn3-017-comparison-20260810T071900Z-e884b7d`.
Their SHA256 values are respectively
`44d4bbf1a64490ee7b4ed697f2bc34a529cb80e403a6ed0177f4b9d06694a6d8`,
`cf76bb6a28668272897784cdef798fc71d75371e5ff57814df2230e29d188c4e`,
and `211c426da8c619ee93c3ad0caeca501d65eb225026899b9fa408dc869b6ac87e`.

## 9. Decision

Discard P-GDN3-017. The strict implementation and one-step production path are
valid, so this is not a kernel, migration or optimization-connectivity failure.
The fixed eight-slot Raven control plane learns a high-amplitude allocation,
then approaches single-slot selection without increasing hard-board closure.
Content-dependent allocation over the existing K rows is therefore not the
missing mechanism at this parent and budget.

Do not run slot-count, top-k, temperature, route-scale, initialization, seed,
LR, loss, batch, width/depth or duration rescue. No automatic successor is
authorized. A future Raven/GDN hybrid must add a qualitatively different,
stable recurrent state organization rather than reparameterize this softmax
router.
