# WordPiece GDN2 FutureSeed Depth Gate

- Plan: `P-CAUSAL-020`
- Status: preregistered
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
