# P-REPRO-001: Same-Seed Native GDN2 Replay

## 1. Metainfo

- Status: preregistered; implementation complete
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

Pending the single registered GPU run.

## 6. Decision

Pending.
