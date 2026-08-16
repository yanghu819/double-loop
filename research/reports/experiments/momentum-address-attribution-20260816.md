# P-DIAG-MOMADDR-001: P059 Address Versus State Attribution

## 1. Question

For P059's 151 adjacent-owner swaps, does the final-layer query geometry
already prefer the wrong owner's key, or does the query prefer its correct key
while the recurrent state returns the wrong value?

## 2. Fixed Diagnostic

Load the exact P059 checkpoint and frozen 1,000-case L1024 test bank. Replace
only the Python reference to the external chunk operator with a transparent
recorder that saves its exact corrected Q and K inputs, then calls the same
operator unchanged. Normalize the recorded tensors with the pinned FLA
normalizer used by the kernel. No tensor returned to the model is modified.

For every query and both layers, compare its own write key with the other
three owners. Record the own-minus-best-other margin, fraction of heads that
rank the own key first, and, for valid-value swaps, the margin to the owner of
the predicted value. The replay must match all 4,000 formal predictions
exactly. This diagnostic has no training, model selection or quality arm.

## 3. Registered Decision

Open a distinct address-organization branch only if, in layer 1:

- at least 60% of swaps have negative own-versus-predicted-owner margin;
- the correct-event median margin exceeds the swap median by at least `.02`;
- the highest-similarity competing owner equals the predicted-value owner in
  at least 50% of swaps.

Open a committed-state-edit transition branch if at most 30% of swaps have a
negative own-versus-predicted-owner margin. Intermediate evidence authorizes
no architecture until a more specific counterfactual is registered.

No result may reopen direct decoupled keys, bounded tangent keys, dense
Log-SPD, stable-token anchors, generic preconditioners, cache/replay, Raven
routers, state-width or training sweeps.

## 4. Status

Registered before the exact pushed-source GPU replay. Pending result.
