# WordPiece GDN2 FutureSeed Data-Diversity Gate

- Plan: `P-CAUSAL-021`
- Status: completed; data-diversity mechanism signal passed, strong gate missed
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

The preregistered preflight and formal run used detached GitHub SHA
`916197899e4405764636ced8406e1dc34db902cf`. The source parquet produced the
registered 160,000 windows after 319,488 rows and 94,986,651 UTF-8 bytes. The
prepared training tensor hash was
`74efb7e6bef6e05cd284f985249e55e81736c89ae3a3f8715faea924982c8562` in
both preflight and formal execution.

Preflight verified one A100 GPU1, 463 byte-identical official FLA files, four
`ChunkGDN2FunctionBackward` layers, Triton Q/K/V convolutions, identical
parameters and initialization, scale-zero hidden/output difference exactly
zero, causal future dependency exactly zero, FutureSeed dependency `0.65193`,
and finite nonzero gradients for all three seed routes.

| Readout | Causal L4 | FutureSeed L4 | FS - causal |
|---|---:|---:|---:|
| masked accuracy | 0.275833 | 0.292493 | +0.016660 |
| masked CE | 5.300604 | 5.131930 | -0.168674 |
| exact 128-token windows | 0 | 0 | 0 |
| suffix-removal CE cost | 0 | 0.689943 | +0.689943 |
| input tokens | 20.48M | 20.48M | 0 |
| parameters | 4,965,722 | 4,965,722 | 0 |
| warmed train throughput | 455,073 tok/s | 455,140 tok/s | +0.01% |
| peak training allocation | 2.042 GB | 2.062 GB | +19.7 MB |

The masked-accuracy delta has paired-window 95% interval
`[0.00973, 0.02364]`; the CE advantage interval is
`[0.14567, 0.19203]`. The same sign is already present at step 1000, where
accuracy delta is `+0.01371` and CE advantage is `0.16003`.

Relative to P020's repeated-window protocol, causal accuracy changes by
`-0.00084` while FutureSeed accuracy changes by `+0.00822`. The matched
accuracy gap therefore grows by `+0.00907`, narrowly below its `+0.01`
diagnostic. More importantly, the CE gap grows from `0.09948` to `0.16867`, a
`+0.06919` gain that passes the preregistered `+0.05` data-diversity gate.
The FutureSeed suffix-removal cost also grows from `0.58652` to `0.68994`.

## Visual Audit

Across all 4,742 masked targets, FutureSeed repairs 182 causal errors and
regresses 103 causal-correct targets, producing 79 net repairs. P020 produced
147 repairs, 111 regressions and only 36 net repairs. The new data regime thus
more than doubles net hard decisions rather than merely lowering probability
loss.

Repairs are balanced across the sequence (`84` left half, `98` right half), as
are regressions (`56/47`). At window level there are 84 repair-only, 39
regression-only, 42 mixed and 91 changed-but-still-wrong cases. The strongest
windows remove three errors with no regression; the worst window adds three.
The archived same-window HTML includes 12 repair, regression and hard-error
cases, while `diagnostics.json` records all aggregate and hardest-case counts.

## Decision

P021 supports the narrow data-scaling mechanism: repeated exposure to a tiny
independent corpus was materially suppressing FutureSeed's real-text value.
At identical 20.48M-token compute, 16x more independent windows increases CE
advantage by `0.06919`, gives a fully positive accuracy interval and more than
doubles net repairs.

It still misses the paper headline gate: accuracy delta is below `0.03`, CE
advantage is below `0.20`, and exact remains zero. Therefore do not claim a
competitive masked-language model yet. The registered positive endpoint slope
and data-diversity pass authorize one clean next experiment that scales both
independent data and total training tokens. Do not run a data-size table,
second seed, or rescue the current endpoint with a few extra steps.
