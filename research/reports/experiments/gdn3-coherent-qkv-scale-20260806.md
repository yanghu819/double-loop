# P-GDN3-003: Coordinate-Coherent FutureSeed/GDN3

## Status

- Status: preregistered; CUDA contract pending
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
