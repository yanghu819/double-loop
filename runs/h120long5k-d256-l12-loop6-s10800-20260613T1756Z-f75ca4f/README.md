# h120 +5k Hard-Stage Attempt Abort

- Run: `h120long5k-d256-l12-loop6-s10800-20260613T1756Z-f75ca4f`
- Machine: AIStation GPU1 only
- Source SHA: `f75ca4f911f505957709b6572b0ff3fc180edf62`
- Status: aborted before first checkpoint eval

## What Happened

This was the original +5k h120 hard-stage extension attempt: same clean
D256/L12/loop6/all-loop setup, but with `108-120:5000` and total step `10800`.

AIStation GPU1 hit its lease limit and halted before the first checkpoint eval
could be written. The last recorded training log was still in the early
curriculum, so this run is not score-bearing evidence for or against the
mechanism.

## Decision

Do not interpret this run. It exists only as provenance for the restart. The
lease-aware replacement is:

`h120long4k-d256-l12-loop6-s9800-20260613T1943Z-f75ca4f`

That replacement answered the same high-value question within the renewed lease
by testing the +4k h120 hard-stage point.

## Artifacts

- `abort.json`
- `config.json`
- `source_HEAD.txt`
- `source.patch`
- `source_snapshot.tar.gz.sha256`
- `source_snapshot.ls.txt`
- `logs/run.log`

The full `source_snapshot.tar.gz` exists on GPU1 under the run directory but was
not committed because it is about 149 MB.
