# P-DIAG-MOMQCF-001: Momentum Query-Owner Counterfactual

## 1. Question

Does the frozen P059 Momentum state contain the correct value under the exact
write address even when the learned query points elsewhere, or did the live
state transition bind the wrong content to that address?

## 2. Intervention

Replay the exact P059 checkpoint and 1,000-case L1024 test bank. At each of the
four MQAR query tokens, replace only the corrected Q entering a selected
Momentum scan with the corrected K from that query's own write token. Test the
first layer, final layer, and both layers separately. The external Momentum
kernel, K/V updates, state, logits head and all weights remain unchanged.

This is an oracle counterfactual used only to identify a causal bottleneck. It
is not a deployable mechanism and its scores are not formal quality claims.
Baseline replay must match all 4,000 frozen predictions exactly.

## 3. Registered Decision

Open a distinct owner-address organization only if the final-layer
counterfactual repairs at least 50% of the 151 swaps, retains at least 95% of
the 3,777 already-correct events, and improves total accuracy by at least
`.015`. Open a committed-state-edit transition if no nonbaseline intervention
repairs more than 10% of swaps. Intermediate evidence opens no architecture.

No result may reopen direct decoupled keys, bounded tangent keys, dense
Log-SPD, generic preconditioners, Raven routers, replay/cache, state-width or
training sweeps. The oracle address itself may not enter a formal model.

## 4. Status

Completed from exact pushed/read-back source
`87874ada218ebd1f3def949fef8a6eee6a733a1f` on the sole A800 CUDA device.
The replay matched all 4,000 baseline predictions exactly and exited zero.

## 5. Result

| condition | balanced | future | past | retained correct | repaired swaps |
| --- | ---: | ---: | ---: | ---: | ---: |
| native P059 | `.94425` | `.95150` | `.93700` | `1.00000` | - |
| layer 0 own key | `.02050` | - | - | - | `.01325` |
| layer 1 own key | `.49025` | `.04700` | `.93350` | `.51496` | `.09272` |
| both layers own key | `.02200` | - | - | - | `.01325` |

The strongest intervention repairs only `14/151` original wrong-key swaps and
creates 1,832 new errors. It misses every owner-address opening gate and
passes the committed-state-edit opening gate because no intervention repairs
more than 10% of swaps.

The direction split is decisive. The final layer's exact local write key
preserves past retrieval but nearly eliminates future retrieval. Its learned
query is therefore specifically co-adapted to decode the incoming producer
Momentum state. The remaining tail is not explained by a query that merely
missed the correct local key; the state content and live edit semantics are
causal.

## 6. Integrity And Cost

- result SHA256:
  `3dcc33d311687df3c365eb930cd6b80378db7ac4f560c2cdd33a076864d51929`
- GPU sample SHA256:
  `d04edaffd1cb562a1ad4b750c41c5eba3ea9d2022c23cbf11852ea82d455f7fb`
- source snapshot SHA256:
  `849b074d9087f077c7f927da34ea9f5e463a43b98a6348c019f3075f6f6c07be`
- run:
  `/huyang2/double-loop/runs/p-diag-momqcf-001-20260816T004339Z-87874ad`
- 67 GPU samples total; 42 active-memory samples; active utilization mean
  `26.81%`, peak `85%`; peak memory `3,014 MiB`; peak power `245.56 W`.

## 7. Decision

Close direct owner-key substitution, nearest-key alignment and decoupled-key
interpretations. Open exactly one distinct family: a modification to the live
committed Momentum transition that can change stale post-commit residual in
`M` without adding another address wrapper or side memory. The oracle key may
not enter that mechanism.
