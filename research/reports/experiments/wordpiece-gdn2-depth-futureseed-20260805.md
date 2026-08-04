# WordPiece GDN2 FutureSeed Depth Gate

- Plan: `P-CAUSAL-020`
- Status: completed; valid weak signal, depth-amplification gate missed
- Date: 2026-08-05 CST
- Machine: AIStation task-mode GPU1 only

## Question

Does generic model-depth scaling convert FutureSeed's proven right-context
signal into materially better real-text masked-token decisions?

## Mechanism Hypothesis

In `P-CAUSAL-019`, a D128/L2 official-FLA GDN2 with native FutureSeed had one
cross-layer route: layer 0's terminal recurrent state initialized layer 1. The
route was real: removing the suffix worsened FutureSeed CE by `0.43915`, while
the strict-causal arm changed by exactly zero. Yet FutureSeed improved endpoint
CE by only `0.08735` and accuracy by `0.00253`.

The clean first-principles explanation is insufficient downstream compute, not
a missing path. A D128/L4 stack creates three consecutive terminal-state
transfers and two additional generic recurrent transformations. If depth is the
missing scaling axis, the matched FutureSeed advantage should grow substantially
without changing data, state width, objective, tokenizer or training tokens.

## Fixed Protocol

- Use the exact offline checkpoint, tokenizer, WikiText JSON, manifests, hashes,
  frozen tied lexical table and validation corruption from `P-CAUSAL-019`.
- Use D128/L4/H4/D32, expand-v1, short-conv4 and the pinned official FLA GDN2
  chunk/Triton path. Depth is the only scaled model axis.
- Compare strict-causal `future_seed_scale=0` with native terminal-state
  `future_seed_scale=1`. Both arms have identical parameters, initialization,
  data order, optimizer and compute; FutureSeed adds no layer or extra pass.
- Train each arm on exactly 20.48M input tokens: batch 128, 1,250 steps and the
  same 160,000 unique `(corruption, window)` pairs used by P019.
- Record endpoint and step0/250/500/750/1000/1250 masked CE, accuracy and exact;
  paired case intervals; future dependency; suffix utility; parameters, tokens,
  throughput and memory; and same-window repair/regression visualization.
- Require exactly three active cross-layer seed routes. Every used seed gate
  must receive a finite nonzero gradient and report per-layer gate, source RMS
  and normalized seed norm.

## Prediction And Decision

Strong support requires the carrier to open and FutureSeed to improve masked
accuracy by `>=0.03` or CE by `>=0.20`, with a paired 95% interval above zero,
the same sign at step1000, and a positive suffix-utility interval. This supports
the claim that generic depth scaling turns FutureSeed's cheap future-context
route into useful real-text predictions.

A final CE gain `>=0.15` that exceeds the L2 advantage by `>=0.05` is recorded
as depth amplification, but remains below the strong paper gate if neither
primary threshold passes. Anything weaker means the L2 probability signal does
not scale enough at this budget.

## Kill Criteria

- Stop before training for any wrong GPU, dirty source, asset/hash drift,
  fallback, non-official FLA path, unequal parameter/init/data state, nonzero
  strict-causal future dependency, scale-zero identity failure, nonfinite CUDA
  forward/backward, or fewer than three active nonzero-gradient seed routes.
- Stop the scientific line if the causal carrier does not open, or if L4 misses
  both the strong gate and the registered depth-amplification diagnostic.
- Do not rescue with longer training, L3/L6/L8, LR, mask, tokenizer, width,
  freeze policy, objective or seed sweeps.

## Claim Boundary

A strong pass supports one narrow claim: increasing generic recurrent depth
amplifies native FutureSeed's future-context utility on an established
real-text masked-token carrier. It does not establish pretrained-BERT parity,
language-model superiority, asymptotic efficiency, or bidirectional-attention
quality parity.

## Execution

Preregistered implementation commit:
`43fa6129fbe309ac5733ba4ca3f1eb06c44e583b`.

```bash
PREFLIGHT_ONLY=1 \
RUN_NAME=wordpiece-gdn2-depth-fs-preflight-20260804T2315Z-43fa612 \
WORDPIECE_CONFIG=configs/retrieval/wordpiece_gdn2_depth_futureseed.env \
./scripts/run_wordpiece_gdn2_futureseed.sh

RUN_NAME=wordpiece-gdn2-depth-fs-formal-20260804T2320Z-43fa612 \
WORDPIECE_CONFIG=configs/retrieval/wordpiece_gdn2_depth_futureseed.env \
./scripts/run_wordpiece_gdn2_futureseed.sh
```

The GPU1 preflight completed at `2026-08-04T23:11:55Z`. It verified the exact
single A100 UUID, 463 byte-identical official FLA files, four
`ChunkGDN2FunctionBackward` layers, Triton Q/K/V short convolutions, scale-zero
hidden/output identity, causal future dependency zero, FutureSeed dependency
`0.65157`, and three active gate gradients of
`0.00963/0.00658/0.00482`.

The formal run completed at `2026-08-04T23:16:25Z` with exit status 2 only
because the registered scientific threshold missed. GPU memory returned to
zero and no process remained.

## Results

| Readout | Causal L4 | FutureSeed L4 | FS - causal |
|---|---:|---:|---:|
| masked accuracy | 0.276677 | 0.284268 | +0.007592 |
| masked CE | 5.323694 | 5.224210 | -0.099484 |
| exact 128-token windows | 0 | 0 | 0 |
| future dependency | 0 | 0.497879 | +0.497879 |
| suffix-removal CE cost | 0 | 0.586523 | +0.586523 |
| total / trainable parameters | 4,965,722 / 1,058,906 | same | 0 |
| input tokens | 20.48M | 20.48M | 0 |
| peak training allocation | 2.040 GB | 2.059 GB | +18.9 MB |

The causal carrier opened and lowered CE by `5.1622`, so this is a valid model
comparison. The accuracy delta has paired-window 95% interval
`[0.00084,0.01426]`, and the CE improvement interval is
`[0.07965,0.11964]`. Both effects are positive at steps 1000 and 1250.

Depth made the weak effect more reliable but did not amplify it materially.
Relative to P019 L2, accuracy advantage grows by `0.00506`, while CE advantage
grows by only `0.01213`; the preregistered depth requirement was at least
`0.05`. The strong `+0.03` accuracy and `0.20` CE gates both miss.

All three trained routes remain active. Their final gates are
`0.4782/0.4912/0.4729`, with normalized seed norms
`15.30/15.72/15.13`. Suffix removal hurts CE by `0.58652`, larger than L2's
`0.43915`. More layers therefore consume more right-context information, but
the marginal information does not translate proportionally into endpoint
quality.

## Visual Audit

The same-window visualization is at
`runs/wordpiece-gdn2-depth-fs-formal-20260804T2320Z-43fa612/visualizations/index.html`.
Across all 4,742 masked targets, FutureSeed repairs 147 causal errors and
regresses 111 correct causal predictions, leaving 36 net repairs. There are 73
repair-only, 47 regression-only, 39 mixed, 96 changed-but-still-wrong and one
stable window. Repairs are balanced across the sequence halves (`69/78`), as
are regressions (`58/53`). The best selected windows remove three net errors;
the worst add three. This is broader than the L2 `119/107` split, but remains
far from a qualitative change in behavior.

## Decision

`P-CAUSAL-020` falsifies the simple hypothesis that one cross-layer transfer is
the main reason P019 was weak. L4 makes the top-1 gain statistically positive
and increases measurable right-context use, but the CE benefit barely grows and
misses both the strong and depth-amplification gates.

Stop L3/L6/L8 and longer-step rescue runs. Retain P019/P020 as real-text
mechanism evidence: FutureSeed carries useful suffix information at constant
sequence-state complexity, but shallow depth scaling alone does not make it a
competitive masked-language model. A future language result needs a genuinely
larger pretraining regime or a different established carrier, not another
small-depth point.
