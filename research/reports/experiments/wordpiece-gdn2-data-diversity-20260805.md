# WordPiece GDN2 FutureSeed Data-Diversity Gate

- Plan: `P-CAUSAL-021`
- Status: preregistered
- Date: 2026-08-05 CST
- Machine: AIStation task-mode GPU1 only

## Question

Was the weak real-text FutureSeed gain in P019/P020 limited by repeatedly
remasking a very small independent corpus, rather than by the future-context
mechanism itself?

## Data Audit

P019/P020 used a train JSON containing the first 20,000 WikiText source rows.
It has 12,894 nonempty rows, 5,880,678 UTF-8 bytes and only 10,003 grouped
length-128 WordPiece windows. The 20.48M reported input tokens were produced by
16 different corruptions of the same first 10,000 windows. They were therefore
20.48M optimization tokens but only about 1.28M independent text tokens.

The locally downloaded and then uploaded full raw parquet is pinned at SHA256
`3136309f1626dd348aaa0b8ab9e4c8a319d175c0a6223e392d60575083603a42`.
It contains 1,801,350 rows, 1,165,029 nonempty rows and 539,295,549 UTF-8
bytes. The fixed P018 validation source and corruption remain unchanged.

A data-only parity check compares the first 20,000 parquet rows with the exact
P019/P020 JSON source. Row text, the first 1,280,000 token IDs and every
special-token-mask bit are exactly equal. Their token tensor SHA256 is
`8313809fd2dd1b7bcc82ba4b7072aa4062cce667ff9711977863e8c646c79381`.
The new source therefore extends the old preprocessing order rather than
silently changing tokenization semantics.

## Mechanism Hypothesis

Repeated corruptions of the same 10,000 windows let both models repeatedly fit
the same local lexical patterns. That can reduce the marginal value of the
suffix summary even when diagnostics prove the summary is used. If independent
context diversity is the missing scaling axis, FutureSeed should gain more
than the causal control when the model sees 160,000 distinct windows once,
without receiving more optimizer steps or tokens.

## Fixed Protocol

- D128/L4/H4/D32 official-FLA GDN2 with the exact P020 native terminal-state
  FutureSeed implementation and three cross-layer routes.
- Same frozen tied BERT-Tiny WordPiece table, fresh common MLM transform,
  parameters, initialization, optimizer, seed, batch and Triton kernel.
- Same 15% standard MLM corruption and exact P018 validation tensors.
- Same 1,250 updates, batch 128 and 20.48M input tokens per arm.
- Sole intervention relative to P020 data: `10,000 windows x 16 corruptions`
  becomes `160,000 windows x 1 corruption`.
- Compare strict-causal scale 0 against native FutureSeed scale 1. No reverse
  scan, extra pass, task rule, selector, search, repair or oracle state.

## Prediction And Decision

The existing strong paper gate remains masked-accuracy delta at least `+0.03`
or CE advantage at least `0.20`, with paired 95% lower bound above zero,
positive sign at step 1000 and positive FutureSeed suffix utility.

The narrower data-diversity hypothesis passes if the FutureSeed CE advantage
grows by at least `0.05` over P020's `0.09948`, or the accuracy advantage grows
by at least `0.01` over P020's `0.00759`, while all integrity and carrier gates
pass. A mechanism pass below the strong gate authorizes one later clean run
that scales both independent data and total compute. A miss closes fixed-compute
data diversity as the primary explanation.

## Budget

One GPU CUDA preflight and one sequential matched pair, at most two wall-clock
hours. The formal model work remains 20.48M input tokens per arm; CPU data
tokenization is measured separately and is not presented as model throughput.

## Kill Criteria

- Stop before training on wrong GPU UUID, extra visible GPU, dirty source,
  parquet/validation/tokenizer/checkpoint hash drift, unofficial FLA source,
  fallback, unequal parameter/init/data state, nonzero scale-zero dependency,
  dead FutureSeed route or nonfinite CUDA backward.
- Stop the line if the causal carrier does not reach accuracy `0.10` and a CE
  drop of `0.50`, or if neither registered data-diversity amplification
  threshold passes.
- Do not rescue with another window count, corruption count, epoch, LR, seed,
  width, depth, mask rate, tokenizer, objective or freeze-policy point.

## Claim Boundary

A pass supports only this claim: at fixed optimization compute, broader
independent language data makes the generic FutureSeed right-context route
more useful. It does not establish pretrained-BERT parity, bidirectional
attention parity, language-model superiority or asymptotic speed.

## Results

Pending the preregistered GPU1 run.
