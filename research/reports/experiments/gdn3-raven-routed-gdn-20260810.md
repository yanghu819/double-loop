# P-GDN3-017: Raven-Routed GDN

## 1. Metainfo

- Status: preregistered; awaiting the admitted single-A100 40GB task
- Date: 2026-08-10
- Branch: `codex/gdn3-raven-routed-update-20260810`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: one AIStation task-mode A100 40GB only
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

## 8. Required Readout

Report mixed and official51-55/56-60/61-64 loop1-5 exact, blank accuracy and
wrong cells; train CE and same-board correction; allocation deviation,
slot/board/token variation, normalized entropy, min/max allocation, routed K/g
relative change, router weight RMS and terminal-state geometry; independently
warmed throughput, peak allocated/reserved memory and timing stability; source,
parent, checkpoint, metrics, config and log hashes; and same-board loop1-5
visualization.

## 9. Decision

Pending strict GPU1 contract, exact step3001 production probe and the single
matched step3100 endpoint. No nearby rescue is preregistered.
