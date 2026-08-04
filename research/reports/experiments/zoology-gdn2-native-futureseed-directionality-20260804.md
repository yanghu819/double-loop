# Native FutureSeed Directionality In Validated Zoology GDN2

- Plan: `P-CAUSAL-007`
- Status: preregistered
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only

## Question

Does native FutureSeed turn a causal GDN2 stack into a cheap route for future
information, or is the strong Sudoku result specific to that task?

## Native Mechanism

This experiment uses the frozen FutureSeed contract, not a reverse scan. For
two causal GDN2 layers, layer 0 scans the sequence once and returns its terminal
recurrent matrix `S_0^T`. Before layer 1 starts its own left-to-right scan, the
state is normalized per example and head and gated:

```text
rms_h = sqrt(mean(S_0^T[h]^2))
seed_h = scale * sigmoid(gate_h) * S_0^T[h] / max(rms_h, 1e-6)
S_1^0 = seed
```

At `scale=0`, layer 1 receives no initial state and is strictly causal. At
`scale=1`, early layer-1 tokens can read associations summarized by layer 0
after layer 0 has seen the full input. There is no right-to-left recurrence,
bidirectional concatenation, oracle state, selector, search, or repair.

## Mechanism Hypothesis

A random value written after its query is information-theoretically unavailable
to a strictly causal model at that query position. The same GDN2 model should
solve a write-before-query example. Native FutureSeed should preserve the past
case and open the query-before-write case because the later layer starts from
the earlier layer's terminal associative memory.

## Fixed Task

Use the P-CAUSAL-006 validated upstream Zoology data/trainer/loss/metric shell
with one deterministic directional MQAR generator:

- vocabulary 256, sequence length 64, four unique key/value associations;
- 10,000 train examples and 1,000 fixed validation examples;
- every example has two `past` associations whose writes precede their queries
  and two `future` associations whose queries precede their random writes;
- key, value, and filler vocabularies are disjoint, preventing accidental
  value leakage through filler tokens;
- query-only cross entropy and per-example accuracy, with a `direction` slice.

Both arms use D128/L2/H4/D32 official FLA GDN2, expand-v1, short-conv4,
training chunk recurrence, BF16 layer autocast, learned positions, the same MLP,
batch 32, AdamW LR 1e-3/WD 0.1/cosine, seed 123, and at most 20 epochs. The only
experimental variable is `future_seed_scale=0` versus `1`. The per-head gate is
present in both arms, so parameter count and initialization are matched.

## Prediction

- no-FutureSeed: past accuracy opens, future accuracy stays near random chance;
- FutureSeed: past remains open and future accuracy rises sharply;
- a valid mechanism result requires the FutureSeed path to carry gradients and
  scale 0 to be equivalent to the validated causal GDN2 output path.

## Budget And Kill Criteria

- One full-batch GPU1 preflight and one sequential matched pair, at most 30
  wall-clock minutes total.
- Stop before training on wrong GPU count/UUID, source SHA, FLA class/path,
  non-chunk training recurrence, non-Triton short convolution, scale-0 output
  mismatch, non-finite forward/backward, dead FutureSeed gradient, or fallback.
- Stop and audit leakage if no-FutureSeed future accuracy exceeds 0.10 while
  past accuracy is open.
- If neither arm reaches past accuracy 0.90, classify the directional task or
  wrapper as invalid and do not interpret FutureSeed.
- Do not rescue through a second seed, LR/width/depth/epoch/loss sweep, reverse
  scan, selector, search, repair, or easier vocabulary.

## Success And Claim Boundary

Strong support requires:

- no-FutureSeed past accuracy at least 0.90 and future accuracy at most 0.10;
- FutureSeed future accuracy at least 0.90 and at least +0.80 over no-FS;
- FutureSeed past regression at most 0.05;
- matched initialization, data hashes, optimizer, training budget, and strict
  official FLA kernel provenance.

Partial support is FutureSeed future accuracy at least 0.50 with delta at least
+0.40 and preserved past accuracy. Anything weaker is not sufficient for a
cheap-bidirectionality claim. This experiment can establish a cross-task causal
mechanism result; it cannot by itself establish language-model quality.

## Pre-Run Harness Correction

The first attempted pair used separate past and future Zoology DataSegments.
Zoology iterates segments sequentially, so each epoch trained all past batches
and then all impossible-future batches. The no-FutureSeed arm ended at chance on
both past (`0.012`) and future (`0.0105`), violating the carrier gate. Its exact
PGID was terminated before interpreting the FutureSeed arm, and `abort.json`
records the failure.

This is a batching bug, not a model result. Before the rerun, the generator was
changed to put two past and two future associations in every example. Every
batch is therefore balanced while the total examples, four associations,
vocabulary, sequence length, loss, model, optimizer, and ten-epoch budget stay
fixed. Directional metrics are now computed from disjoint preregistered query
position bands. No model or loss hyperparameter changed.
