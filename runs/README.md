# Archived Run Evidence

This directory is an append-only experiment archive, not a list of maintained
entry points. A run directory may contain a successful result, a failed
hypothesis, or an aborted infrastructure attempt.

Only runs named from the canonical configs in `../configs/sudoku/` should be
used as current baselines. Check `../docs/DEPRECATIONS.md` and each run's
`config.json`, `score.json`, source SHA, and report before citing it.

Checkpoints, datasets, and large model artifacts are intentionally excluded
from Git.

When several arms use the same exact source SHA, one aggregate run may retain
the full `source_snapshot.tar.gz`; each arm then carries a
`source_snapshot.ref` with the shared path and SHA256. This avoids committing
identical 60+ MiB snapshots once per arm without weakening provenance.
