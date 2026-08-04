# WordPiece GDN2 FutureSeed Real-Text Gate

- Plan: `P-CAUSAL-019`
- Status: completed; valid weak signal, strong gate missed
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

Implementation commit: `ebed1755ac24194144c2d5105e3e86a2229e4295`.

The first strict GPU preflight was:

```bash
PREFLIGHT_ONLY=1 \
RUN_NAME=wordpiece-gdn2-fs-preflight-20260804T2235Z-ebed175 \
./scripts/run_wordpiece_gdn2_futureseed.sh
```

It correctly stopped before training with status 1 and `abort.json`: upstream
Zoology constructs token embeddings on CUDA by default, while the fixed BERT
anchor had been returned to CPU. A direct `torch.equal` therefore compared two
devices before the already device-independent byte hashes ran. The corrective
commit compares detached CPU tensors and changes no model, data, optimization,
metric or registered decision gate.

The second preflight at corrective SHA `d0c030e1` also stopped before training.
Its autograd auditor stored only Python object IDs while traversing the graph;
released node wrappers allowed ID reuse, so it incorrectly stopped after
`NllLossBackward0 -> LogSoftmaxBackward0` and reported that the deeper FLA node
was absent. The follow-up keeps node wrappers alive during traversal. This
changes only the fail-closed auditor, not model execution or the protocol.

The corrected preflight and formal run used detached source
`4296f20ed37e19dbe5b14d5c17deca3cdaf611cf`:

```bash
PREFLIGHT_ONLY=1 \
RUN_NAME=wordpiece-gdn2-fs-preflight3-20260804T2245Z-4296f20 \
./scripts/run_wordpiece_gdn2_futureseed.sh

RUN_NAME=wordpiece-gdn2-fs-formal-20260804T2251Z-4296f20 \
./scripts/run_wordpiece_gdn2_futureseed.sh
```

## Artifacts And Result

The first preflight implementation abort is at
`runs/wordpiece-gdn2-fs-preflight-20260804T2235Z-ebed175`; its prepared schedule
hash is `99fb776e5703376411b72c861bd368557228e4f40d4369385c6c5223b4231a26`.
The second implementation abort is at
`runs/wordpiece-gdn2-fs-preflight2-20260804T2240Z-d0c030e` and has the same
prepared schedule hash.

The passing preflight is
`runs/wordpiece-gdn2-fs-preflight3-20260804T2245Z-4296f20`. It verified one
GPU1, the exact BERT carrier anchor, 463 byte-identical FLA wheel/install files,
Triton Q/K/V convolutions, `ChunkGDN2FunctionBackward`, scale-0 output identity,
zero causal future dependency, nonzero FutureSeed dependency and gate gradient,
equal parameters/init, and frozen tied lexical tensors.

The formal run is
`runs/wordpiece-gdn2-fs-formal-20260804T2251Z-4296f20`. It exited with status 2
only because the preregistered strong scientific gate missed; all integrity and
carrier gates passed.

## Results

| Readout | Causal GDN2 | GDN2 + FutureSeed | FS - causal |
|---|---:|---:|---:|
| masked accuracy | 0.276466 | 0.278996 | +0.002531 |
| masked CE | 5.350222 | 5.262873 | -0.087349 |
| exact 128-token windows | 0 | 0 | 0 |
| future dependency | 0 | 0.406000 | +0.406000 |
| suffix removal CE cost | 0 | 0.439155 | +0.439155 |
| trainable / total parameters | 561,418 / 4,468,234 | same | 0 |
| input tokens | 20.48M | 20.48M | 0 |
| peak training allocation | 1.666 GB | 1.672 GB | +6.30 MB |

The causal carrier opened at step 250 and lowered CE by `5.1556`, so this is a
valid model comparison. The FutureSeed CE advantage was `0.08620` at step 1000
and `0.08735` at step 1250. Its paired-window CE 95% interval is
`[0.06723, 0.10778]`, so the small probability-quality gain is statistically
stable. It misses the registered `0.20` strong threshold. Accuracy improved by
only 12 of 4,742 masked targets; its interval `[-0.00335, 0.00851]` crosses
zero and misses the `+0.03` threshold.

The mechanism diagnostic is much stronger than the endpoint gain. Replacing
the suffix changes causal CE by exactly zero, but worsens FutureSeed CE by
`0.43915`, with bootstrap interval `[0.32460, 0.56772]`. The trained seed gate
is `0.4823`. FutureSeed therefore genuinely reads useful right context on real
text; the present two-layer shell does not convert most of that information
into better top-1 token choices.

## Visual Audit

The matched visualization is at
`runs/wordpiece-gdn2-fs-formal-20260804T2251Z-4296f20/visualizations/index.html`.
Across all 256 windows, FutureSeed repairs 119 masked tokens and regresses 107,
for only +12 net. There are 53 repair-only, 51 regression-only, 38 mixed, 111
changed-but-still-wrong and 3 stable windows. The effect is not confined to one
sequence side: left/right repairs are 60/59 and regressions are 50/57. Selected
cases include clean three-token repairs and a four-token regression, preventing
the aggregate CE gain from hiding unstable top-1 behavior.

## Decision

Classify P-CAUSAL-019 as a valid weak mechanism transfer, not a paper headline
pass. It supports the narrow statement that native terminal-state FutureSeed
delivers useful right-context information in a matched real-text causal GDN2.
It does not support language-model superiority, BERT parity, or a cost claim.

Do not rescue this exact L2 run with more steps, LR, mask, tokenizer, freeze
policy or seed. The next high-information language experiment, if pursued,
should test one preregistered depth scale: more recurrent layers create more
cross-layer state-transfer opportunities. That asks whether the observed
right-context CE signal scales into decisions, rather than tuning this endpoint.
