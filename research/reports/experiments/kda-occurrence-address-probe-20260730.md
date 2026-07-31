# KDA Occurrence-Address Causal Probe

## 1. Metainfo

- Plan: `P-OCC-001`
- Date: 2026-07-30 CST
- Machine: AIStation GPU1 only
- Source branch: `codex/gdn2-address-binding-20260730`
- Status: implementation complete; CUDA smoke retry pending

## 2. Hypothesis

KDA's channel-wise decay can preserve multiple time scales, but repeated writes
to one semantic key still share an address. Decay therefore represents age, not
the queryable statement "the Nth write to this key." If address collision is
causal, a parameter-free occurrence-index rotation applied to KDA Q/K should
sharply improve repeated-key retrieval.

This is deliberately separate from Sudoku. Sudoku cells have unique absolute
addresses; repeated-key occurrence is a different memory capability.

## 3. Configuration

- Task: nth-from-start repeated-key retrieval.
- Every sample contains 128 key-value writes and one external query.
- The queried key appears exactly 8 times during training.
- Both arms receive identical additive key, value, type, and occurrence
  embeddings.
- Baseline: content-addressed official KDA.
- Candidate: identical model and parameters, plus fixed occurrence-index rotary
  binding on write K and query Q.
- Official pinned FLA `chunk_kda` remains the recurrent state update.
- One seed; both arms begin from the exact same state dict and consume the same
  batches.
- Primary OOD evaluation: length 512, queried key repeated 16 times.

## 4. Environment

- Persistent root: `/huyang2/double-loop`
- CUDA: GPU1 only, exactly one visible UUID
- FLA source: `9c8e42e762fce087c27b673af4922795d9edb85e`
- Backend dispatch disabled; no CPU model path and no alternate kernel fallback

## 5. Commands

```bash
GPU1_UUID=<current-gpu1-uuid> CUDA_VISIBLE_DEVICES=0 \
  scripts/run_kda_occurrence_probe.sh smoke

GPU1_UUID=<current-gpu1-uuid> CUDA_VISIBLE_DEVICES=0 \
  scripts/run_kda_occurrence_probe.sh formal
```

## 6. Artifacts

Pending. Each run will contain `config.json`, `score.json`, `curves.json`,
`cases.json`, `index.html`, source snapshot, launch environment, log, source
SHA, and GPU UUID.

## 7. Results

GPU1 became available on 2026-07-31 11:18 CST. The first CUDA smoke stopped
before model construction because the launcher looked for the pinned FLA
installation under the detached worktree's private `.cache`. The audited
installation actually lives in the shared persistent project cache:
`/huyang2/double-loop/.cache/fla-active`.

This is an environment failure, not an experiment result. The launcher now
uses `/huyang2/double-loop/.cache` for every detached worktree and the runtime
provenance check reads the source marker from that same root. No package was
downloaded, no kernel fallback was enabled, and no CPU model path was used.

## 8. Decision

- Continue to a learned key-local counter only if the oracle candidate improves
  OOD accuracy by at least `+0.20` and reaches at least `0.70`.
- Reject the direction if the OOD gain is below `+0.10` after 300 steps.
- An intermediate result is analyzed once, not followed by beta, frequency,
  seed, or width sweeps.
- Do not implement Versioned KDA first. It changes state capacity and kernel
  structure, so it is a later upper bound rather than the causal test.

## 9. Submission

Not applicable.
