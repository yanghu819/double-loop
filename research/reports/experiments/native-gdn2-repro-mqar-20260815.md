# P-REPRO-001: Same-Seed Native GDN2 Replay

## 1. Metainfo

- Status: complete; reproducibility gate passed
- Task: directional MQAR L1024, four future and four past queries
- Model: D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed
- Fixed endpoint: two sequential 10-epoch/batch32/seed123 native replays
- Purpose: validate the measurement protocol before another GDN3 mechanism

## 2. Evidence And Hypothesis

Recent nominally identical native controls with the same initialization and
data hashes have landed from roughly `.04` to `.48` balanced accuracy. The
within-run control remains the correct comparator, but this spread is large
enough that a small architecture delta could reflect BF16/Triton optimization
bifurcation rather than a causal mechanism.

The falsifiable question is whether two byte-identical same-seed native runs in
one process produce decision-equivalent predictions. This is not a seed sweep,
model smoke, quality arm or hyperparameter experiment. It changes no model,
data, objective, budget or kernel.

## 3. Exact Protocol

Serialize one native initialization. Run `native_replay_a` and
`native_replay_b` sequentially through the existing `run_arm`; each loads that
same file, sees identical data and warmup batches, and calls
`set_determinism(seed)` after warmup immediately before `Trainer`.

## 4. Registered Gate

Integrity requires exact initialization/parameter count, identical data and
warmup hashes, one active native FutureSeed route and pinned official kernels.
Decision stability requires balanced delta `<=.01`, each directional delta
`<=.015`, joint delta `<=.01`, total-error delta `<=40`, and at least `.90`
query-level prediction agreement. Replay-B post-warm wall must be `<1.25x`.

If all pass, proceed to one new live memory organization and continue using
strict contemporaneous controls. If any quality-reproducibility check fails,
do not interpret architecture deltas below `.05`; first register a deterministic
training protocol. No seed, precision, duration or hyperparameter rescue is
authorized by this diagnostic.

## 5. Results

The single registered A100-40GB run completed with status 0 from exact pushed
and read-back source `2791a7f44fe7f861aa2bbaa9ed78e881d48aca0e`.
Both arms loaded initialization hash
`3e8fe038f1401735168783c2de1d9217a8807e88685ee12010142fc7b05e4c44`,
used identical train/test/warmup hashes, and produced the same trained-parameter
hash `fa28ed30664a92d3536ed52132fb70887550fec5f2176f9e11711e20c04898b3`.

Replay A and B are exactly equal on every quality endpoint:

- balanced/future/past accuracy: `.494/.454/.534` for both;
- joint exact: `.041` for both;
- total errors: `2024` for both;
- query-level prediction agreement: `4000/4000 = 1.0`;
- wrong-key valid-value swaps: `1546` for both, or `.763834` of errors;
- all registered quality deltas are zero.

Replay B post-warm wall is `.715503x` replay A, below the fixed `1.25x` gate.
Independent warmed throughput is `1822.83/1782.59` examples/s for A/B and peak
CUDA allocation is identical at `1,084,782,080` bytes. During training the
small D128/L2 model used about `4.6-4.8 GiB`; sampled utilization averaged
about `52.9%` and reached `78%`, with one compute application on the required
GPU UUID. No NaN, OOM or fallback occurred.

The decision JSON SHA256 is
`df8ae1c212590666440f12029d60c74a6d09cbc059b23e58bcf864c00ae98854`.
The arm checkpoint SHA256 values are
`e44f5e2866b42976c7a926517981e571452d794ed7df975b486d4f7df16a12e2`
and `e35c61928583f5630c45c8477bbcfa0d38cdc1441eb0b08407020787707c2adf`;
the serialized containers differ, while the trained parameter hash is exact.
Formal-log/source-snapshot/parent-init SHA256 values are
`c86da25585d3f94265cdfe86a6b220bdb5f7d8b0a6dcf86accd30233bb75aca9`,
`6c2da1f9328e7041bcfff333334e3f4ae079acf0db6a69ea57c8f1b0fe896df2`,
and `7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850`.

## 6. Decision

Pass. The same-seed native protocol is decision-stable, so the recent spread
across different matched runs is not evidence of uncontrolled within-protocol
BF16/Triton nondeterminism. Continue to require a contemporaneous native
control, but architecture deltas can be interpreted under this protocol.

The endpoint also sharpens the mechanism target: `76.38%` of errors retrieve a
valid value under the wrong key. The next GDN3 successor must change how a
committed key-value pair occupies live recurrent memory, preserving explicit
ownership under interference. Another dense companion state, Q/K wrapper,
payload code, pre-scan residual or Raven-style controller is not authorized.
