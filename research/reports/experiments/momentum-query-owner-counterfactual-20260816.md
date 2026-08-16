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

Registered before the exact pushed-source GPU replay. Pending result.
