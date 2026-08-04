# WordPiece GDN2 FutureSeed Real-Text Gate

- Plan: `P-CAUSAL-019`
- Status: approved; not launched
- Date: 2026-08-05 CST
- Machine: AIStation task-mode GPU1 only

## Question

On an independently validated real-text right-context task, does native
FutureSeed improve an otherwise matched strictly causal official-FLA GDN2?

## Mechanism Hypothesis

P-CAUSAL-018 established that the exact WordPiece/WikiText evaluation contains
a large right-context signal: one official BERT-Tiny checkpoint lost `0.1398`
masked accuracy and `1.4207` CE when only its attention mask became causal.
P-CAUSAL-007 established that native FutureSeed can carry an earlier recurrent
layer's terminal state into the next layer and expose future writes without a
reverse scan. If that route is not specific to synthetic key/value layouts, it
should reduce masked-token CE on this real-text carrier.

The official BERT checkpoint supplies only one common frozen lexical table:
`bert.embeddings.word_embeddings.weight`. It is tied to the output decoder in
both arms. Position embeddings, the dense-GELU-LayerNorm MLM transform, output
bias, residual MLPs and both recurrent cores are freshly and identically
initialized. The BERT encoder and its contextual states are discarded. This
avoids relearning a 30,522-word lexical geometry inside the small budget, but it
is not a pretrained-BERT mixer replacement and cannot support that claim.

## Fixed Protocol

- Assets: exact P-CAUSAL-018 checkpoint revision/hash, P-CAUSAL-009 WordPiece
  tokenizer, WikiText train/validation JSON and manifests, all offline.
- Data: first 10,000 of the available 10,003 grouped length-128 train windows.
  Sixteen deterministic
  standard 15% MLM corruptions are generated once and then reused by both arms.
  The validation set is the exact P-CAUSAL-018 seed-123 corruption over 256
  grouped windows and 4,742 masked targets.
- Model: D128/L2/H4/D32, expand-v1, short-conv4, official FLA GDN2 chunk/Triton.
- Lexical initialization: exact official BERT word table, frozen and tied to a
  fresh decoder in both arms. Every trainable contextual parameter is fresh, so
  both use the already validated GDN2 AdamW `1e-3`, weight decay `0.1`, cosine
  decay without warmup, BF16 and seed 123.
- Budget: batch 128, exactly 1,250 optimizer steps, 20.48M input tokens per arm.
  The 16 ordered corruption banks contain exactly 160,000 distinct
  `(corruption, window)` pairs. Batches cross bank boundaries, so every pair is
  consumed exactly once with no last-batch wrapping or omission.
- Intervention: `future_seed_scale=0` versus `1`. Every trainable tensor exists
  in both arms and starts byte-identically. No loop, extra layer, extra pass,
  reverse scan, attention, oracle, selector, search, repair or text rule.
- Readouts: masked CE/accuracy/exact at step 0/250/500/750/1000/1250, future
  dependency and suffix utility, FS gate gradient, throughput, peak memory,
  paired window bootstrap intervals and same-window token cases.

## Prediction

The strict-causal arm should exceed `0.10` masked accuracy and lower CE by at
least `0.50` from step 0, proving that the recurrent shell opened. Strong
FutureSeed support requires final accuracy `>=+0.03` or CE `>=0.20` better than
causal, a paired 95% interval above zero, the same effect sign at step 1000,
and positive suffix-utility interval. CE gain `[0.05,0.20)` without the accuracy
gate is only a weak mechanism signal, not a headline pass.

## Kill Criteria

- Stop before training on any wrong GPU, asset/source/hash mismatch, server
  download, fallback, dirty source, unequal parameter/init/data/order state,
  nonzero scale-0 future dependency, zero FS dependency/gate gradient, nonfinite
  CUDA forward/backward, or scale-0 identity failure.
- If the causal arm finishes below `0.10` or lowers CE by less than `0.50`,
  classify the recurrent shell as unopened;
  do not use the comparison as FutureSeed evidence.
- If FutureSeed misses strong support, archive a weak `[0.05,0.20)` CE signal
  separately; otherwise close this protocol. Neither outcome gets an LR,
  tokenizer, mask, width, depth, epoch, loss, freeze-policy or seed rescue.
- Sequential first-use Triton compilation makes raw wall time diagnostic only.

## Claim Boundary

A pass supports: native terminal-state FutureSeed improves matched causal GDN2
masked-token recovery on one established real-text carrier while preserving a
linear recurrent state path. It does not support: pretrained BERT replacement,
general language-model superiority, bidirectional-attention parity, or an
efficiency claim.

## Commands

To be filled after the implementation commit and strict GPU preflight.

## Artifacts And Result

Not launched.
