# P-GDN3-003: Coordinate-Coherent FutureSeed/GDN3

## Status

- Status: discarded at preregistered step500 carrier gate
- Date: 2026-08-06
- Benchmark: official/full-diversity 9x9 Sudoku, 51-64 blank scaling cliff
- Compute: AIStation task-mode GPU1 only
- Seed: 52 only

## Decision Context

P-GDN3-001 established a real cross-layer compatibility signal: a shared
address projection improved mean hard blank accuracy from `0.2037` to `0.4479`
at matched step9100. P-GDN3-002 then falsified the scalable form of that idea.
One namespace shared by all twelve layers produced genuine loop correction but
ended at official 51-55/56-60/61-64 exact
`0.3008/0.0996/0.0762`, hard macro `0.1589`, and mixed exact `0.2168`.

The failed endpoint rules out a shared-namespace strength, rank, seed, width,
or duration rescue. It does not rule out the compatibility mechanism itself.

## Read-Only Geometry Audit

Before selecting a new mechanism, adjacent-layer Q/K/V geometry was measured
on the frozen canonical step9000 and step12000 checkpoints and the P-GDN3-002
step12000 checkpoint. For each adjacent layer and head, the audit compared the
raw bases with a joint Q/K orthogonal Procrustes map.

Representative canonical step12000 measurements:

- normalized K identity residual: about `1.422`;
- normalized K residual after optimal joint Q/K Procrustes: about `1.108`;
- normalized distance of the fitted rotation from identity: about `1.413`;
- minimum cross-basis singular value: about `0.006`.

The same pattern appears at canonical step9000 and P-GDN3-002 step12000. Mature
adjacent bases are not small perturbations of one another. Even the best
orthogonal bridge leaves a large residual and contains nearly null directions.
This rejects a post-hoc free K/V rotation as the next experiment; it would
repeat the destructive degrees of freedom already seen in P-FS2-001.

## Mechanism

`coherent_qkv` changes initialization only:

1. Construct the ordinary D256/L12 official-FLA GDN2 stack.
2. Copy layer0 Q, K, and V projection modules into every deeper layer.
3. Copy layer0 Q, K, and V short-convolution modules into every deeper layer.
4. Keep every parameter in distinct storage and optimize every layer
   independently from the first step.
5. Leave output projections, decay/erase/write dynamics, LayerNorm, FFN,
   FutureSeed RMS normalization, and per-head FutureSeed gates layer-private.

There is no new parameter, recurrent call, scan direction, task rule, loss,
regularizer, selector, or inference-time operation. FutureSeed still seeds layer
`l+1` from layer `l`'s terminal recurrent state using the canonical per-example
and per-head RMS normalization and learned head gate.

## Mechanism Hypothesis

FutureSeed needs a common state coordinate system early enough for the whole
stack to co-adapt. Permanent mature sharing is too restrictive, while a learned
post-hoc rotation is too ill-conditioned. Exact coordinate coherence at birth
should let transported state become useful during optimization; independent
parameters should then recover the layer-private capacity lost by P-GDN3-002.

## Predictions

- Initial Q/K/V projection and short-convolution states are bit-identical across
  all layers, but their parameter pointers are distinct.
- At least one optimizer step makes those modules measurably diverge, proving
  that this is initialization rather than hidden weight tying.
- Parameter count and inference graph match ordinary D256/L12 GDN2+FutureSeed.
- Early Sudoku opening should be no worse than P-GDN3-002, while the hard tail
  should improve because deeper layers are free to specialize.
- If coordinates decorrelate too quickly for co-adaptation, the registered
  step3000 or step6000 gates will fail and close the mechanism without rescue.

## Fixed Training Contract

- D256/L12/H8/K32/V32, channel multiplier4, 11.49M-class parameters;
- twelve pinned official-FLA GDN2 chunk/Triton layers;
- native FutureSeed scale1, unit RMS normalization, per-head learned gate;
- full-diversity 12k curriculum, random traversal, loop5, all-loop CE;
- microbatch32, gradient accumulation4, effective batch128;
- AdamW contract and LR/weight decay copied from P-GDN3-002;
- seed52 and one trajectory only.

The only intended difference from normal GDN2+FutureSeed is the construction-
time Q/K/V coordinate copy. The only intended difference from P-GDN3-002 is
removing its permanent shared-address residual and using coherent initialization.

## CUDA Contract

The full-size experiment is authorized only if GPU1 validation proves:

- one visible CUDA index0 with UUID
  `GPU-53e9f3b4-2966-65d3-6614-09c540921519`;
- pinned FLA source and Triton convolution backend;
- exact initial equality of Q/K/V projection and short-convolution state;
- distinct parameter storage across layers;
- non-coordinate output projections remain private;
- official `ChunkGDN2FunctionBackward` remains in the graph;
- finite forward/backward and nonzero coordinate gradients;
- coordinate modules diverge after one optimizer step;
- finite D256/L12 forward/backward without fallback or OOM.

### Contract Result

Passed on task-mode GPU1 from clean detached source
`ccd889798ecdfc9e114014699087de7bcc9ac7af`:

- visible CUDA device: `NVIDIA A100-SXM4-80GB`, index0, UUID
  `GPU-53e9f3b4-2966-65d3-6614-09c540921519`;
- pinned FLA source SHA:
  `9c8e42e762fce087c27b673af4922795d9edb85e`;
- initial coordinate max-absolute error: `0.0`;
- parameter-count delta versus independent initialization: `0`;
- private output-projection difference: `0.246773`, so non-coordinate
  modules were not accidentally copied;
- official backward graph contains `ChunkGDN2FunctionBackward` and
  `CausalConv1dFunctionBackward`;
- one optimizer step changes coordinate parameters by `1.904e-5` and creates
  cross-layer divergence `1.809e-5`, with finite nonzero Q gradients in every
  sampled layer;
- full D256/L12 contract forward/backward is finite with `11,459,136`
  reasoner parameters and sampled batch1 peak allocation/reservation
  `365.35/376.00 MiB`.

The separate two-step batch32 run through the production runner also passed:
the complete Sudoku model has `11,485,760` parameters, wrote a valid checkpoint
and evaluator bundle, and used peak allocation/reservation
`12,832.52/13,568 MiB`. Its zero exact score is expected at step2 and is not a
science result.

Archived engineering evidence is under
`research/reports/visualizations/gdn3-coherent-qkv-scale-20260806/`. The remote
source snapshot is retained at the fit run directory with SHA256
`d62a371c6432fedbc937743b231ea5752d1c15442e63a49d94c8d7a4884e6af4`.

## Formal Trajectory

The sole registered trajectory launched at `2026-08-06T11:10:38Z`:

- run: `gdn3-coherent-qkv-d256l12-s12000-20260806T111038Z-ccd8897`;
- run directory:
  `/huyang2/double-loop/runs/gdn3-coherent-qkv-d256l12-s12000-20260806T111038Z-ccd8897`;
- source: clean detached SHA
  `ccd889798ecdfc9e114014699087de7bcc9ac7af`;
- process group: `125922`; Python child: `125990`;
- launch log:
  `/huyang2/double-loop/artifacts/launch/p-gdn3-003/formal-ccd8897.log`.

The run cleared first-shape compilation and remained finite, but failed the
first registered science gate. At frozen step500, holes50 loop5 exact/blank is
`0/0.1321`; loops1-5 exact are all zero and blank accuracy remains
`0.1319/0.1305/0.1317/0.1325/0.1321`. Train CE is `2.0089`. The matched
P-GDN3-002 readout at the same step is exact/blank `0.7778/0.9853` and CE
`0.0143`.

The completed gate was detected after the process had reached step800. The
supervisor wrote `abort.json` and sent SIGTERM only to formal PGID `125922`;
the process group exited and GPU allocation returned to zero. There was no
NaN, OOM, fallback, source drift, or infrastructure fault. This is a clean
mechanism failure: exact coordinate equality at birth creates a symmetric deep
stack that does not specialize quickly enough to open even the easy carrier.
No seed, LR, loss, width, initialization-strength, or partial-sharing rescue is
authorized.

## Science Gates

- **Step500:** holes50 loop5 exact `>=0.60`, finite CE, intact source/data/GPU
  provenance.
- **Step3000:** holes53 loop5 exact `>=0.015625` and blank `>=0.5907`, with
  positive loop correction. This must at least match P-GDN3-002 at the same
  step rather than merely produce nonzero exact.
- **Step6000:** holes53 exact `>=0.05` or holes58 exact `>=0.01`, CE below
  P-GDN3-002's `0.8711`, and genuine loop-wise wrong-cell reduction.
- **Endpoint:** mean official 51-64 exact `>=0.3191` or mixed exact `>=0.40`,
  plus genuine loop correction and no systems regression relative to the
  unchanged normal-GDN2 graph.

## Kill Criteria

Stop and write `abort.json` on CUDA UUID, source SHA, data hash, metric, or
checkpoint mismatch; parameter aliasing; failure to specialize; fallback, OOM,
NaN, or non-finite gradients; or any registered science-gate miss. Do not rescue
with another seed, LR, loss, batch, width, partial sharing, initialization
strength, training duration, or nearby mechanism.

## Allowed Claim

A pass would support the claim that coordinate-coherent initialization is a
scalable FutureSeed/GDN3 interface: it gives cross-layer recurrent state a common
starting language without imposing a permanent shared namespace. A miss closes
this initialization hypothesis only; it does not negate the sealed causal value
of native FutureSeed or justify a post-hoc basis-rotation sweep.
