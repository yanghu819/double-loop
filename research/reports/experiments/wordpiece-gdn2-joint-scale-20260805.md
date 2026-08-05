# P-CAUSAL-022: Real-Text Joint Data and Compute Scale

- Status: preregistered, not yet run
- Date: 2026-08-05 CST
- Resource: AIStation task-mode GPU1 only

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
