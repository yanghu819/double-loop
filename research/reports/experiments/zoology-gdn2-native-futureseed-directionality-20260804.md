# Native FutureSeed Directionality In Validated Zoology GDN2

- Plan: `P-CAUSAL-007`
- Status: done
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
batch 32, AdamW LR 1e-3/WD 0.1/cosine, seed 123, and exactly 10 epochs. The only
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

## Result

The corrected, balanced run completed on 2026-08-04 using one task-mode GPU1.
It passes every preregistered strong-support condition:

| Metric | No FutureSeed | FutureSeed | Delta |
|---|---:|---:|---:|
| past accuracy | 0.9305 | 0.9955 | +0.0650 |
| future accuracy | 0.0110 | 0.9930 | +0.9820 |
| past exact | 0.8770 | 0.9910 | +0.1140 |
| future exact | 0.0000 | 0.9860 | +0.9860 |
| balanced accuracy | 0.47075 | 0.99425 | +0.52350 |
| past CE | 0.29065 | 0.01479 | -0.27586 |
| future CE | 4.57885 | 0.01943 | -4.55941 |

The control is a valid carrier rather than a failed model: it reaches 93.05%
on causally available past queries. It remains at 1.10% on future queries,
close to the 1/96 random-value rate. FutureSeed solves both directions and does
not trade away the causal direction.

The training curves separate immediately. FutureSeed balanced validation
accuracy is 0.4900 after epoch 0 and 0.9605 after epoch 2; no-FutureSeed is
0.01025 and 0.01875 at the same points and reaches only 0.47075 by epoch 9.

## Integrity And Systems Diagnostics

- identical parameter count: 538,704 in both arms;
- identical initialization hash:
  `1bbba1ac6d1bd2edcb4f85e47805c7662756da53f413cb7582e8061c46e18835`;
- identical train/test hashes:
  `31bac228...d6d5bf9` and `3fa26a5a...470209`;
- scale-0 wrapper versus validated upstream GDN2 max output difference: 0;
- changing later values changes no-FS early-query logits by exactly 0;
- the same perturbation changes FS early-query logits by mean 0.07178;
- FutureSeed gate gradient max: 5.64e-4, so the route participates in training;
- learned gate: 0.5262; normalized seed-state norm: 16.84;
- strict pinned official FLA GDN2 chunk recurrence and Triton short convolution;
- one visible A100 GPU; GPU returned to 0 MiB after completion.

Raw fixed-budget measurements are 703.7 versus 935.7 examples/s and 350.5
versus 343.2 MB peak allocated memory. They are archived but are not evidence
of a FutureSeed speedup: the no-FS arm ran first and paid one-time Triton
compilation. A randomized or separately warmed systems protocol is required
for an efficiency claim.

No outer reasoning loop is used here, so loop1-to-final gain is not applicable.
This experiment deliberately isolates cross-layer terminal-state transfer.

## Decision

`P-CAUSAL-007` strongly supports the narrow cross-task mechanism claim:
FutureSeed gives a causal recurrent stack access to information that appears
later in the input, without a reverse scan and without increasing parameters.
Together with the matched Sudoku frontier result, this is now evidence on two
tasks. It still does not establish language-model quality or general-purpose
bidirectional replacement quality.

Do not repeat this result with another seed, easier vocabulary, or a length
table. The next paper gate must test transfer: a single preregistered OOD
length/association generalization result or an established language/retrieval
task where FutureSeed is compared with causal GDN2 and a full noncausal
reference under an honest compute protocol.

## Artifacts

- Run: `runs/zoology-gdn2-fs-directionality-mixed-20260804T0427Z-181896e`
- Comparison: `output/comparison.json`
- Same-sequence cases: `output/paired_hardest_cases.json`
- Visualization: `visualizations/index.html`
- Source snapshot SHA256:
  `a23059a9cebd969df20913d0128e650ec19aecc0f86bb173fdf52f47ebc93157`
