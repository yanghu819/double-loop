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

Complete and discarded as a successor source. R1 stopped before Python/CUDA because a
full-repository source archive spent several minutes materializing 340 MiB on
the shared filesystem and the supervising process sent the registered PGID a
termination signal. The runner wrote a non-science `abort.json`; no metric,
prediction or model forward exists. R2 changes only source-archive scope to
the five registered files and keeps the model, artifacts, intervention and all
gates unchanged. R2 completed normally with status 0 on the sole registered
A800.

## 6. Result

| variant | balanced | future | past | joint | errors | wrong-key swaps | adjacent swaps |
|---|---:|---:|---:|---:|---:|---:|---:|
| native P059 | .94425 | .95150 | .93700 | .82400 | 223 | 151 | 151 |
| output correction off | .56050 | .28350 | .83750 | .05700 | 1758 | 180 | 159 |

Disabling only `q <- q - exp(D)k` loses `.38375` balanced accuracy and `.767`
joint exact. It changes 1,455 formerly correct queries to unrelated wrong
values and another 110 to valid values under the wrong key. Only 23 of the
original 151 wrong-key swaps become correct, while 65 remain wrong-key and 63
become another wrong value. Every registered selection check fails.

The trained parameter hash is unchanged at
`04f716382abd3a45f732105a25f3b39af04a10486d81afc60589cf60a97e01ac`.
The exact P059 checkpoint, score, cases and test-data hashes are respectively
`13410aaa...b8508f`, `7201d328...f6e8a`, `f64b0ae0...26f20` and
`4a8237ba...e278f`.

## 7. GPU And Artifacts

The R2 telemetry contains 69 samples, 18 with active compute. Active samples
average `56.50%` utilization, peak at `79%`, and observed memory peaks at
`3,016 MiB`. This is a frozen diagnostic, not a training throughput number.

- run: `/huyang2/double-loop/runs/p-diag-momread-001-20260816T131629Z-5c60c54`
- diagnostic SHA256: `880fa3650d1f37e02c03795269349ce71a03b97daef3eb83b2b86e414ed40f0b`
- log SHA256: `3ea52222d2b69452beab2d324912fc29a9f433d3d7bd633450f03849152a481d`
- telemetry SHA256: `1ddd8948cae3b8946ae0293502f103f1aae98be268b9d175b2edcb22fe166930`
- source snapshot SHA256: `bc3814e4ee24e19361970f0f6cf791b1de1bafa415609e483f803128df16ba9b`

## 8. Decision

Close removal, rescaling, sign changes, token/head gating and replacement of
the P059 output correction. The correction is a necessary co-adapted read
geometry, not the residual owner-interference cause. The remaining 151-swap
tail is already encoded in the live `[S,M]` trajectory. The next experiment
must preserve Q/K/V convolution and `q-Dk`, and change the scalable live
Momentum state organization itself.
