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

Completed from exact pushed/read-back source
`32f4787d1310a23efdfe11de1dbcb46a5f25fb86`; all 4,000 baseline predictions
replayed exactly and the run exited zero.

In the final layer, 65.56% of swaps prefer the predicted wrong owner's key,
but this is not a clean address discriminator. Correct events have an
own-minus-best-other median of `-.10638`, versus `-.12653` for swaps, making
the registered separation `-.04218` rather than `>=.02`. The nearest competing
key equals the predicted owner in only 31.13% of swaps, below 50%. Moreover,
81.41% of correct events do not rank their own write key first by mean-head
similarity. Raw nearest-key cosine is therefore not the model's operative
owner contract.

The learned key space is severely anisotropic: final-layer mean anisotropy is
`26.22`, effective-rank fraction `.06625`, and condition about `35,440`.
That is real geometry collapse, but it does not selectively explain failures.
The preregistered result is mixed and authorizes no architecture. Proceed only
to the exact own-write-key query counterfactual in `P-DIAG-MOMQCF-001`.

The 71 GPU samples contain 23 active-memory rows: active utilization averages
`41.91%`, peaks at `79%`, and sampled memory peaks at `3,050 MiB`.

Artifacts:

- run: `/huyang2/double-loop/runs/p-diag-momaddr-001-20260816T003324Z-32f4787`
- diagnostic SHA256: `7e4a37939624e500de4d37eadc470cb98e4619ab215e1e21b2d3b07854b7eb1e`
- GPU samples SHA256: `111ff12f61aae51dd55d4975b581c25995d4b8b542a6062e63a44c0bcc349e78`
- source snapshot SHA256: `ef5c828d358c0c93e8bb280b08072ea691a64c7cfe787e0953a357bd61f7a96e`
