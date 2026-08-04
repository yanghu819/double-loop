# Strict GDN2 In Validated Zoology MQAR Shell

- Plan: `P-CAUSAL-006`
- Status: done
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

## Result

The strict GDN2 carrier passed decisively and much earlier than the validated
MHA carrier.

| Mixer | Epoch 0 | Epoch 1 | Epoch 2 | Epoch reaching >0.99 |
| --- | ---: | ---: | ---: | ---: |
| official MHA | 0.246 | 0.250 | 0.259 | 25 |
| strict official GDN2 | 0.242 | 0.984 | 0.9965 | 2 |

GDN2 validation loss moved `2.82 -> 0.0815 -> 0.0222`; upstream early
stopping fired at epoch 2. This is not a small endpoint difference. In the
validated shell, GDN2 reaches the associative-retrieval phase transition in
roughly one tenth as many epochs as the documented MHA baseline.

The run took approximately 91 seconds including process startup and Triton
warm-up. The sampled training VRAM was 3,841 MiB. The separate full-size
forward/backward preflight used 538,696 model parameters, reported 0.328 GiB
peak allocated CUDA memory, and produced finite loss and gradients.

## Integrity

- Project SHA: `906fa19be2236baac25824196a973205902594e2`.
- Zoology SHA:
  `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`.
- Official FLA source SHA:
  `9c8e42e762fce087c27b673af4922795d9edb85e`.
- Runtime class: `fla.layers.gdn2.GatedDeltaNet2` from the pinned FLA tree.
- Training recurrence mode: `chunk`; Q/K/V short convolutions used the
  official `ShortConvolution` modules with `FLA_CONV_BACKEND=triton`.
- Full-size preflight and training both used exactly one visible GPU1.
- Source snapshot SHA256:
  `e3c37e85c30291e409627534ba5d52fd3552c26c98869dc93b6c627ee4c0c898`.
- Exit status was zero; GPU memory returned to 0 MiB.

## Conclusion

The causal GDN2 carrier is not the bottleneck. The prior zero-retrieval result
came from replacing the upstream shell, not from GDN2's memory rule. We now
have the necessary controlled platform for the actual mechanism experiment:
keep this shell and GDN2 fixed, introduce a transparent future-query split,
and compare native FutureSeed against no FutureSeed from matched initialization.
