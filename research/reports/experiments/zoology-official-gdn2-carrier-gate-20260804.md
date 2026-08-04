# Strict GDN2 In Validated Zoology MQAR Shell

- Plan: `P-CAUSAL-006`
- Status: in progress
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only

## Question

Can strict official FLA GDN2 solve ordinary causal MQAR when it is placed in
the exact upstream shell that P-CAUSAL-005 proved can reach 99.775%?

## Mechanism Hypothesis

P-CAUSAL-005 shows that MQAR has a sharp optimization transition and that the
upstream training recipe reaches it. P-CAUSAL-004 stopped before that behavior
because it changed the entire recurrent shell. If GDN2 itself has adequate
causal associative memory, changing only the sequence mixer inside the now
validated shell should recover a similar opening.

## Fixed Contract

- Same Zoology SHA, generator, train/test examples, batch, loss, metric,
  optimizer, schedule, seed, positional embeddings, MLP state mixer, two-layer
  depth, max100 epochs, and `>0.99` early stop as P-CAUSAL-005.
- Only sequence mixer changes from upstream one-head MHA to strict official FLA
  `fla.layers.gdn2.GatedDeltaNet2` at SHA `9c8e42e`, D128/H4/D32,
  expand-v1, short-conv4, training chunk kernel, BF16 layer autocast.
- The adapter only maps Zoology's `d_model` argument and tuple return type; it
  does not alter recurrence, gates, state, logits, labels, or loss.
- No FutureSeed in this run.

## Prediction

If GDN2 is a valid carrier, validation accuracy should exceed `0.99` within the
official 100-epoch budget. Its curve may open at a different epoch than MHA,
but it must show associative retrieval rather than vocabulary-partition loss.

## Budget And Kill Criteria

- One full-size GPU1 forward/backward preflight, then one training run, at most
  30 wall-clock minutes.
- Stop on wrong SHA/GPU count, CPU execution, non-official class, non-Triton
  convolution, NaN/OOM, or kernel/dependency fallback.
- If final accuracy is below `0.90`, classify the GDN2 carrier as closed and do
  not run FutureSeed. Between `0.90` and `0.99`, inspect the curve but do not
  rescue through seed/LR/head/width/epoch sweeps.

## Decision And Claim Boundary

Passing `>0.99` authorizes the first fair no-FutureSeed versus native
FutureSeed comparison in this exact shell. This run alone supports only GDN2
carrier validity; it cannot support a FutureSeed claim.
