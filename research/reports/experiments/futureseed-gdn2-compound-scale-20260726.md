# FutureSeed GDN2 Compound Scale

## Metainfo

- Plan: `P-SCALE-037`
- Status: resource gate active; microbatch geometry amended after measured
  performance cliff
- Preregistered: 2026-07-26 19:43 CST / 2026-07-26 11:43 UTC
- Machine: AIStation GPU1, NVIDIA A800-SXM4-80GB
- GPU2: forbidden
- CPU model smoke: forbidden
- Seed: 52 only

## Question

The strict step-500 carrier gate makes official FLA GDN2 the strongest
finite-step FutureSeed carrier: 46-50-blank loop-5 exact is `0.7969`, ahead of
RWKV7 `0.7285`, KDA `0.6465`, and GDN `0.3633`.

Does that advantage survive useful scaling, and can a larger GDN2 trained on
substantially more independent official boards turn the 51-64-blank
per-cell improvement into nonzero full-board exactness?

## Mechanism Hypothesis

GDN2's stronger finite-budget result is not enough to establish a higher
frontier. At 51-64 blanks it is still zero full-board exact, and its per-blank
accuracy is only slightly above RWKV7.

If the remaining failure is ordinary capacity and data under-training, then a
larger official GDN2 with native FutureSeed should show a delayed but clear
hard-range exact-learning curve as training coverage increases. If loss and
blank accuracy improve while hard full-board exact stays flat, the bottleneck
is global closure rather than insufficient ordinary scaling.

This is one compound scaling experiment, not a width/depth/LR/seed table.

## Configuration

- official FLA `GatedDeltaNet2`, pinned wheel and exact source;
- backend dispatch disabled, official chunk backward, Triton short conv;
- D256, 12 layers, 8 heads, head dimension 32, expand-v 1;
- native terminal-state FutureSeed, scale 1, unit normalization;
- five recurrent loops with equal CE supervision on every loop;
- BF16, AdamW grouped contract, LR `0.0015`, weight decay `0.001`;
- microbatch 32, accumulation 4, effective batch 128;
- official 3.832M-board full-diversity training split;
- 12,000 optimizer steps, up to 1.536M sampled boards;
- curriculum `46-50:500,51-55:3500,51-60:4000,51-64:4000`;
- checkpoints every 100 steps;
- formal eval at steps `500,1000,3000,6000,9000,12000`;
- eval ranges `46-50`, `51-55`, `56-60`, and `61-64`;
- no noise, scratch state, feedback, special margin loss, search, repair,
  selector, rollout oracle, or Sudoku rule.

Canonical files:

- `configs/sudoku/gdn2_scale.env`
- `scripts/run_canonical_gdn2_scale.sh`
- `CUDA_VISIBLE_DEVICES=0 ./run.sh baseline`

## Predictions

1. The larger model may trail initially, but by step 3000 it should show either
   nonzero 51-55 exact or a strong CE/blank-accuracy slope.
2. By step 6000 it should approach or exceed the completed D224 GDN reference:
   51-55 exact `0.3926` and 56-64 exact `0.1270`.
3. More loops must increase full-board exact, not only blank accuracy.
4. If GDN2 is the correct quality carrier, its hard-range slope should justify
   continuing to step 12000 and then running one long no-FutureSeed control.

## Decision And Kill Criteria

Preflight rejects the run if any of these fail:

- GPU1 identity or `CUDA_VISIBLE_DEVICES=0`;
- pinned official FLA source/class;
- official GDN2 chunk backward and Torch output/state/gradient reference;
- initial-state/FutureSeed gradients;
- Triton convolution or no-fallback contract;
- clean detached source, official data identity, or finite-value checks.

The two-step GPU capacity probe uses the full D256/L12 model and the canonical
microbatch geometry.
OOM, nonfinite values, wrong GPU, or low utilization with excessive memory
stops the launch. There is no automatic batch/kernel/model fallback; a resource
change requires an explicit recorded config commit.

Training decisions:

- At step 3000, stop only if 51-55 exact is below `0.02`, 51-55 blank accuracy
  remains below `0.58`, and CE improves by less than `0.03` from step 1000.
- At step 6000, stop if 51-55 exact remains below `0.10`, 56-60 exact remains
  zero, and the checkpoint curves are flat.
- Continue to step 12000 if either 51-55 exact reaches `0.30`, 56-60 exact is
  nonzero, or hard-range exact has a clear positive slope.
- Primary success: 51-55 exact `>=0.50` and 56-64 combined exact `>=0.20`.
- Strong success: mixed exact `>=0.45` or 56-64 exact `>=0.30`.

If step 6000 is flat under these rules, ordinary GDN2 capacity/data scaling is
not the missing mechanism. Do not respond with D288/D320/L14/L16 or LR tables.

## Claim Enabled On Success

Success would promote official GDN2 to the main quality carrier and support the
claim that FutureSeed transfers across recurrent state formulations and
continues to benefit from generic model/data/compute scaling. It would not by
itself prove FutureSeed's causal contribution at long scale; that requires the
conditional matched GDN2 no-FutureSeed control.

## Results

### Strict preflight and rejected execution geometry

At 2026-07-26 19:49-20:00 CST, exact SHA
`973a082212f5bfbac1e07c09ba944a5cf39eae6e` passed the GPU1-only strict
preflight. The pinned FLA wheel and installed files match, backend dispatch is
disabled, all 12 GDN2 layers use `fla.layers.gdn2.GatedDeltaNet2`, short
convolution is Triton, and the official chunk recurrence has maximum
output/state/gradient reference errors `5.99e-5/2.76e-4/2.34e-4`. The shared
shell is identical across all four carriers for all `77/77` common tensors.

The first full-size D256/L12 microbatch-64 probe used the real official GDN2
forward and backward on GPU1, allocated about `30 GiB`, and completed two
optimizer steps without OOM or nonfinite values. Its checkpoint reports
`530.06s` elapsed, which includes the first large-shape kernel compilation.

A second identical warm-cache probe rejected the explanation that compilation
alone caused the slowdown. It reports `99.34s` train time for two optimizer
steps, or `49.67s/step`, while repeated GPU samples showed long idle intervals.
The model has `11.486M` trainable parameters. For comparison, the accepted
D192/L10 GDN2 continuation uses about `5.70s/step`; the modest compute increase
does not explain a nearly nine-fold runtime increase.

This triggers the preregistered low-utilization/high-memory stop. A 12k-step
launch at this geometry would be a compute mistake, not useful scaling.

### Explicit amendment

At 2026-07-26 20:23 CST, the execution geometry changes from microbatch
`64 x accumulation 2` to `32 x accumulation 4`. Effective batch `128`, sampled
boards per optimizer step, model, data, loop count, loss, optimizer, seed, and
all scientific decision rules remain unchanged. This is not an automatic
fallback: the failed geometry and both probe runs remain archived, and the
change is committed before a new detached probe.

The next gate is one warm full-size GPU1 probe at the amended geometry. Launch
the formal 12k-step experiment only if measured steady-state throughput returns
to a useful range and no implementation or memory gate regresses.
