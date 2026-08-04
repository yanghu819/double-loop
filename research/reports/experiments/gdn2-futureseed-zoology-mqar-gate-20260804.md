# GDN2 FutureSeed Upstream Zoology MQAR Gate

- Plan: `P-CAUSAL-004`
- Status: preregistered
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only

## Question

Does native FutureSeed provide useful future information on an established
non-Sudoku recall benchmark, after first proving that the matched causal GDN2
carrier can solve the benchmark's ordinary causal direction?

## Mechanism Hypothesis

Zoology MQAR writes key/value pairs before querying them. A causal GDN2 should
solve this ordinary direction. We retain the exact upstream generator and make
one transparent directionality transform for half of every batch: move the
unchanged query block before the unchanged key/value block. The labels and
tokens are not regenerated or altered.

The causal no-FutureSeed arm should learn the ordinary half but remain at
chance on the future half. Native FutureSeed lets a deeper recurrent layer
start from the previous layer's terminal state, so it can receive evidence
from the later key/value block without a reverse scan or noncausal mixer.

## Fixed Contract

- Upstream data: HazyResearch Zoology commit
  `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`.
- Exact upstream `multiquery_ar` generator: vocabulary 8192, sequence 128,
  eight associations, power-law query gaps, random distractors.
- Direction transform: first half untouched; second half swaps the complete
  query and context blocks while preserving tokens, pair order, and labels.
- Model: strict pinned official FLA GDN2, D128/L4/H4/D32, expand-v1,
  short-conv4, chunk/Triton, BF16, no fallback.
- Shared shell: tied token embedding/readout, no position embedding, two train
  loops and four evaluation loops.
- Optimizer: AdamW, LR 1e-3 cosine to zero, weight decay 0.1, seed52.
- Budget: batch128, fixed eval1024, at most 1000 optimizer steps per arm.
- Arms differ only by native FutureSeed scale 0 versus 1.

## Decision Rules

1. Run no-FutureSeed first. At step500 its ordinary past-query accuracy must
   be at least 0.50. Otherwise the carrier is invalid and the FutureSeed arm
   does not run.
2. Strong support requires FutureSeed future-query accuracy at least 0.70,
   FutureSeed minus no-FutureSeed at least +0.50, and past-query regression no
   worse than -0.05.
3. Partial support requires future-query delta at least +0.20 with past
   accuracy preserved and a positive loop1-to-loop4 correction signal.
4. Reject this cross-task claim if FutureSeed future accuracy is below 0.50 or
   the delta is below +0.20 after the valid-carrier gate.

## Kill Criteria

- Wrong GPU count, non-CUDA execution, wrong Zoology or FLA SHA, fallback
  kernel, non-matching initialization/data hashes, NaN/OOM, or invalid K/V
  targets stops the run.
- No second seed, LR/width/depth/loop/loss sweep, reverse scan, selector,
  search, repair, oracle state, or task-specific rule.
- If no-FutureSeed fails the past-query carrier gate, do not rescue the
  benchmark with more steps or an easier vocabulary.

## Claim Boundary

A positive result would support a narrow claim: terminal-state FutureSeed is a
cheap learned route for future input evidence in a causal recurrent GDN2, on
both hard Sudoku and an established associative-recall benchmark. It would not
establish language-model quality or causal-generation validity.
