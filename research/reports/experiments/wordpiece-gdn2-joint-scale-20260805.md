# P-CAUSAL-022: Real-Text Joint Data and Compute Scale

- Status: completed; both strong paper routes and the scaling gate passed
- Date: 2026-08-05 CST
- Resource: AIStation task-mode GPU1 only
- Source SHA: `4fe0193cdc62963b684fc5da6c878f85f828b426`
- Formal run: `wordpiece-gdn2-joint-scale-formal-20260805T004035Z-4fe0193`

## Decision Question

P-CAUSAL-021 proved at fixed compute that replacing repeated corruptions with
independent text grows the FutureSeed CE advantage from `0.09948` to
`0.16867`. Does ordinary data-and-compute scaling continue that trend far
enough to create a strong real-text result, or does this D128/L4 training shell
saturate below the paper threshold?

## Mechanism Hypothesis

FutureSeed passes a compressed summary of the full sequence from a shallow
recurrent layer into the initial state of the next layer. Learning to encode
and use that summary requires diverse lexical and syntactic contexts. P021
showed that repeated masks over a tiny text subset undertrain this route.
Giving the unchanged model four times as many independent windows and optimizer
steps should improve the generic route without adding a task rule, extra layer,
reverse scan, selector or repair mechanism.

## Registered Protocol

- Exact P021 official-FLA GDN2 D128/L4/H4/D32, 4,965,722 parameters per arm.
- Native FutureSeed scale 0 versus 1; identical initialization and all other
  model tensors.
- Frozen tied official BERT-Tiny WordPiece lexical table.
- Hash-pinned full WikiText-103 raw parquet and fixed P018 validation tensors.
- 640,000 independent length-128 windows, one deterministic 15% corruption.
- Batch 128, 5,000 steps, 81.92M input tokens per arm.
- AdamW `1e-3`, weight decay `0.1`, cosine schedule, no warmup, seed 123.
- Evaluation at steps `0/1000/2000/3000/4000/5000`.
- Official pinned FLA GDN2 chunk/Triton path; no fallback.

## Predictions and Gates

Strong paper support requires an opened causal carrier, positive suffix utility,
a paired 95% lower bound above zero, positive FutureSeed advantage at step4000
and step5000, and either:

- masked accuracy delta at least `+0.03`; or
- masked CE advantage at least `0.20`.

A narrower scaling mechanism signal requires either another `+0.01` accuracy
advantage or `+0.05` CE advantage over P021's `0.01666/0.16867`.

## Budget and Kill Criteria

One CUDA preflight and one formal two-arm run, with a two-hour wall cap. Stop on
GPU UUID, source SHA, data hash, tokenizer, initialization, parameter, official
FLA/Triton, scale-0 identity, dependency, gradient, finite-backward or 640,000
window-count failure. Stop on OOM, NaN or wall timeout by exact PID/PGID and
write `abort.json`.

If neither the strong gate nor the scaling signal passes, close this shallow
real-text shell. Do not rescue with width, depth, learning rate, loss, mask
rate, seed, corruption count or a data-size table.

## Claim Enabled by Success

A pass would support that a native terminal-state FutureSeed route gains useful
right-context quality under standard increases in independent language data and
training compute, while retaining the matched official-FLA recurrent backbone.
It would not by itself establish language-model parity or a systems-cost
frontier against a fully pretrained bidirectional Transformer.

## Integrity Result

The independent CUDA preflight and formal run both passed. The task container
exposed only GPU1 (`GPU-53e9f3b4-2966-65d3-6614-09c540921519`). All 463 files
in the pinned FLA wheel matched the activated source tree; all four recurrent
layers used `ChunkGDN2FunctionBackward`, and every Q/K/V convolution used the
Triton backend. Both arms had 4,965,722 parameters and identical initialization.
Scale 0 reproduced direct causal GDN2 exactly, causal future dependency was
zero, FutureSeed dependency was nonzero, and all three cross-layer gates had
nonzero gradients. The 640,000-window tensor hash was reproduced in preflight
and formal execution.

## Results

| Metric | Causal GDN2 | GDN2 + FutureSeed | Difference |
|---|---:|---:|---:|
| masked accuracy | 0.309363 | 0.373893 | +0.064530 |
| masked CE | 4.618276 | 3.905131 | -0.713145 |
| masked exact window | 0.000000 | 0.000000 | 0.000000 |
| input tokens | 81.92M | 81.92M | matched |
| elapsed including validation | 206.65 s | 207.08 s | +0.43 s |
| warmed train throughput | 499,129 tok/s | 475,322 tok/s | -4.77% |
| peak training allocation | 2.048 GB | 2.067 GB | +18.9 MB |

The paired-window 95% interval is `[+0.05492,+0.07438]` for accuracy and
`[+0.66992,+0.75717]` for CE improvement. Both registered strong routes pass.
Relative to P021, the accuracy advantage grows by `+0.04787` and the CE
advantage by `+0.54447`, so both joint-scaling gates pass as well.

The gap grows with training rather than appearing from a noisy endpoint:

| Step | Accuracy delta | CE advantage |
|---:|---:|---:|
| 1000 | +0.005694 | 0.159744 |
| 2000 | +0.033741 | 0.462154 |
| 3000 | +0.053775 | 0.628654 |
| 4000 | +0.057782 | 0.700920 |
| 5000 | +0.064530 | 0.713145 |

Removing the visible suffix changes causal CE by exactly zero but worsens
FutureSeed CE by `2.12152`, with bootstrap lower bound `1.88127`. Endpoint
FutureSeed dependency is `1.34065`; the three learned gates remain active at
approximately `0.421/0.470/0.440`.

## Visual Audit

Across all 4,742 masked targets, FutureSeed repairs 440 causal errors and
regresses 134 causal-correct predictions, for 306 net repairs. Repairs occur on
both halves of the text window (`248/192` left/right), as do regressions
(`63/71`), ruling out a fixed-position explanation. At window level there are
125 repair-only, 21 regression-only, 82 mixed and 28 changed-but-still-wrong
cases. The best window removes six errors without a regression; the worst adds
three. The HTML includes repair, regression and hard mixed windows from the
same fixed validation set.

## Decision

P-CAUSAL-022 is a strong positive paper result. Independent data plus ordinary
training compute does not merely improve both arms: it expands FutureSeed's
matched advantage enough to pass both preregistered quality routes. The clean
claim is that native terminal-state seeding learns an increasingly useful
right-context route under language scaling in an otherwise causal official-FLA
GDN2 stack.

Do not respond with another 2x/4x token table, seed repeat or model-size sweep.
The next high-information experiment should test whether the same scaling law
persists on a larger established text corpus/model regime or establish a robust
quality-cost frontier against a valid bidirectional model. That experiment
requires a separate preregistration; the current result receives no rescue.
