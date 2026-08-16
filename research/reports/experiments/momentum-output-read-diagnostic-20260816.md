# P-DIAG-MOMREAD-001: Momentum Output-Read Counterfactual

## 1. Question

P059 leaves 151 wrong-key valid-value swaps, all adjacent in write rank and 150
within the same direction. Are those residual owner errors already present in
the recurrent state, or are they introduced only by Momentum DeltaNet's final
per-head output correction `q <- q - exp(D) * k`?

## 2. Frozen Intervention

Load the exact P059 checkpoint and deterministic L1024 test set. Evaluate the
ordinary trained model once, then evaluate the same weights with only
`use_output_correction=False` in both layers. This removes `-exp(D)k` from the
read query while preserving Q/K/V convolutions, all learned parameters, the
complete live `[S,M]` recurrence, native FutureSeed, data and predictions in
the native arm. No parameter is trained or changed.

## 3. Why This Is Decision-Changing

P-DIAG-MOMQCF replaced the receiver query with its own write key and showed
that direct query/key alignment is destructive, but it did not isolate the
small built-in `q-Dk` correction. P051 ruled out a fixed lagged commit and P069
proved the short Q/K convolution is useful. This counterfactual cleanly
separates a final read-phase bias from a wrong owner already committed to live
state.

## 4. Integrity And Selection Gates

Require the exact P059 checkpoint, score, cases, test-data hash, source SHA,
single registered A800, and exact native replay of all metrics and cases. The
toggle must not alter the parameter hash.

Open exactly one from-scratch no-output-correction successor only if every
condition holds:

- balanced accuracy improves by at least `.005`;
- wrong-key swaps fall by at least 20;
- future and past accuracy each regress by at most `.005`;
- joint exact regresses by at most `.01`;
- total errors do not increase.

Any miss closes output-correction removal or replacement. Do not sweep the D
scale, sign, head sharing, token gate, key neighbor, seed, LR, loss, width,
depth or duration. Continue only with a distinct scalable live Momentum state
organization.

## 5. Status

Approved; implementation complete; awaiting exact pushed source and sole-GPU
execution.
