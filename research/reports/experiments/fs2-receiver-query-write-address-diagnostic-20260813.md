# P-DIAG-ADDR-001: Receiver Query/Write Address Mismatch

## 1. Metainfo

- Status: completed; branch decision `kill_address_and_cache_line`
- Task: fixed directional MQAR L1024, 1,000 validation cases
- Checkpoint: completed P-FS2-007 surprise-K16 arm
- Intervention: none; zero parameters, zero training, logits unchanged

## 2. Question

P-FS2-007 improves balanced accuracy to `0.48825`, yet `2044/2047` remaining
errors are valid values retrieved from the wrong key. The diagnostic asks one
narrow question: does a receiver query align poorly with the producer's write
address, while aligning materially better after the same token is reprojected
into the receiver's native K basis?

## 3. Exact Measurement

Run the frozen model's ordinary forward. At the official GDN2 call boundary,
read but do not modify normalized producer K, normalized receiver Q/K, and the
official kernel's actual committed edit. For each labeled query, recover its
four candidate association-write positions. The committed write is the value
token at the dataset's `write_position+1`, not the preceding key token. Compute:

- per-head producer-basis rank: receiver Q dot producer K at committed writes;
- per-head receiver-native rank: receiver Q dot receiver K at committed writes;
- top-1, MRR, target-vs-best-other margin, and producer/receiver target-K cosine;
- target-write surprise from the exact committed-edit Frobenius norm;
- whether the target write is among the producer's top-16 surprise tokens;
- producer and receiver key-Gram anisotropy, effective-rank fraction,
  condition number and coherence on the fixed first 128 validation cases;
- exact target-address survival through later official decay/erase transitions,
  plus matched surprise-K16 versus recency-K16 survival on those 128 cases;
- correctness, direction, and wrong-key/valid-value status.

The first batch is run once without hooks and once with hooks; logits must be
bitwise identical. Parameter SHA must be unchanged and every parameter gradient
must remain `None`. No optimizer, backward, checkpoint write, alternate logits,
or model smoke is allowed.

## 4. Outputs

Report overall/future/past/correct/error/swap/high-surprise/low-surprise top-1,
MRR, margins and surprise coverage; Pearson correlations of surprise, address
survival and rank gain with error; wrong-key swap fraction; key geometry;
surprise/recency survival; wall time and peak allocation versus one ordinary
full validation pass. Archive aggregate JSON and event-level JSON. The existing
bit-exact Sudoku P-BIND-001 result remains the fixed loop-side evidence:
surprise-correction AUROC/lift `0.5230/1.0524` and position-QK effective-rank
fraction/anisotropy/condition medians `0.4792/6.4568/402.91`. This run completes
the missing valid L1024 carrier measurement; it does not rerun Sudoku.

## 5. Cost Contract

One warmup, one ordinary 1,000-case validation pass, and one hooked 1,000-case
pass on a single A10080. Full address ranking uses all 1,000 cases; the explicit
FP32 state-survival replay is fixed to the first 128 cases. Because this is
instrumentation rather than a production mechanism, the implementation ceiling
is preregistered at diagnostic/ordinary elapsed `<=3.00x` and peak allocation
`<=1.50x`. A cost miss invalidates this implementation but does not answer the
mechanism question.

## 6. Fixed Branch Rule

- **Receiver-native address/binding:** wrong-key swaps are at least 70% of
  errors, and receiver-native top-1 or MRR improves by at least `0.10` both
  overall and within swaps.
- **Receiver-native surprise cache:** the address rule passes; target writes
  selected by top-16 surprise are at least `0.10` more frequent in errors than
  correct events; and their receiver-native MRR gain exceeds low-surprise writes
  by at least `0.05`.
- **Live-state address interference:** absent receiver-basis mismatch,
  `-survival` predicts errors with AUROC at least `0.70`, correct writes retain
  at least `0.10` more address norm than error writes, receiver anisotropy
  median is at least `4`, and effective-rank fraction median is at most `0.60`.
- **Loop/readout convergence:** receiver-native top-1 is at least `0.80`
  overall and `0.60` on errors while token accuracy remains below `0.85`.
- **Kill address/cache line:** none of the above. Do not launch an address
  wrapper, ridge seed, cache, K/temperature/seed sweep, or Sudoku graft.

The diagnostic alone never admits a model run. Any selected successor still
requires its own exact intervention, prediction, budget, contract, and kill
gate.

## 7. Results

R1 (`p-diag-addr-001-20260814T041429Z-004dd05`) exited before model
construction because the launcher omitted the external Zoology repository from
`PYTHONPATH`; `ModuleNotFoundError: zoology.data.utils` was raised with zero GPU
memory allocated. The launcher wrote `abort.json`. This is an orchestration
failure and carries no mechanism evidence.

R2 fixes only that deterministic environment defect: it adds the fixed
`/huyang2/double-loop/repos/zoology-official` source root, verifies Zoology SHA
`1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`, and invokes the explicit base
Python. It loaded the exact checkpoint and reached the first hooked production
batch, then stopped before any aggregate measurement because the observer
incorrectly required three `chunk_gdn2` captures. Production uses two L1024
chunk scans plus one K16 replay; official FLA dispatches that short replay via
its recurrent path, so the chunk-only address/edit hooks correctly expose two
records. The generated `abort.json` is a second non-science observer-wiring
failure.

R3 changes only the observer contract. Every batch must expose exactly the two
L1024 producer/receiver chunk scans. The recomputed committed-edit top-16 set
must exactly equal the production chronological selection; producer selected
evidence and receiver replay input must be the same tensor object and bitwise
equal; receiver replay terminal/seed RMS and board variation must be finite and
nonzero. The completed P-FS2-007 strict contract remains the independent proof
of exactly two main plus one replay official backward paths. The checkpoint,
data, measurements, cost ceiling and branch rule remain unchanged.

R3 (`p-diag-addr-001-r3-20260814T043416Z-0509af7`) completed with status zero
on the exact clean detached source `0509af7c6f22407b8fd0fedb675e8101d790fdff`.
All integrity checks passed: 1,000 validation cases and 4,000 labeled queries,
bitwise-identical logits, unchanged parameter SHA, no gradients, exactly two
instrumented L1024 chunk scans, exact production top-16 selection/wiring, and
an active finite K16 receiver replay state. The diagnostic JSON SHA256 is
`2e0370b815024cec02f02aecc22cded4f439d4ee12b5b3dc1a0d4a1aebce48b3`.

The model accuracy is `0.48825`; `2044/2047` errors (`0.998534`) are valid-value
wrong-key swaps. Reprojection is not the missing operation: receiver-native
top-1 improves only `0.00725` overall and falls `0.001954` on errors; MRR gains
are `0.006062` and `0.000295`. Direction matters (`+0.12325` future versus
`-0.10875` past top-1), but this asymmetry does not separate correct from error
cases.

Surprise does not identify overwritten evidence. Target writes are already in
the top-16 surprise set for `0.99425` of all queries, slightly more often for
correct than error cases (`0.997952` versus `0.990718`). Surprise-K16 address
survival is `0.040271`, below recency-K16 `0.250834`; correct-minus-error target
survival is `-0.000986`, and survival error AUROC is only `0.527174`. All
registered correlations with error are near zero.

The useful measurement is severe geometry collapse: producer/receiver median
anisotropy is `28.6008/30.3852`, effective-rank fraction
`0.056146/0.042308`, condition number `3678.35/44778.27`, and coherence
`0.99039/1.00066`. Geometry is pathological, but the frozen branch rule
required predictive write loss or receiver-basis improvement; neither occurs.
Therefore the formal branch is `kill_address_and_cache_line`. Do not launch a
surprise-cache, receiver-reprojection, K/admission, Log-SPD-plus-cache, ridge,
or nearby address-wrapper rescue from this checkpoint. A successor must test a
different live recurrent-transition hypothesis with its own preregistration.

Instrumentation cost remained valid: diagnostic/ordinary elapsed `1.4513x`
and peak allocation `1.2536x`, below the frozen `3.00x/1.50x` ceilings.

## 8. Conclusion

The remaining MQAR errors are binding failures, but neither target-write
survival nor receiver-basis reprojection predicts those failures. The evidence
rejects cache admission and static receiver-address repair as the next causal
intervention. Keep the geometry-collapse observation as a mechanistic clue;
move the next experiment into the live state transition and preserve coherent
erase/write/read addressing. No Sudoku transfer is authorized by this result.
