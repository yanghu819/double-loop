# KDA Occurrence-Address Causal Probe

## 1. Metainfo

- Plan: `P-OCC-001`
- Date: 2026-07-30 CST
- Machine: AIStation GPU1 only
- Source branch: `codex/gdn2-address-binding-20260730`
- Status: completed; preregistered mechanism gate rejected

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

- Formal run:
  `runs/kda-occurrence-formal-20260731T032536Z-4b94220`
- CUDA smoke:
  `runs/kda-occurrence-smoke-20260731T031927Z-4b94220`
- Both contain `config.json`, `score.json`, `curves.json`, `cases.json`,
  `index.html`, source snapshot, launch environment, log, source SHA, and GPU
  UUID.

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

The retry passed official KDA forward, terminal-state layout, real backward,
optimizer update, and artifact generation. The 300-step formal run then
completed on GPU1:

| Evaluation | content | occurrence rotary | delta |
|---|---:|---:|---:|
| length128, repeat8 accuracy | 0.0449 | 0.0977 | +0.0527 |
| length128, repeat8 CE | 3.4521 | 3.3617 | -0.0904 |
| length512, repeat16 accuracy | 0.0371 | 0.0313 | -0.0059 |
| length512, repeat16 CE | 3.4974 | 3.4931 | -0.0043 |

The candidate's only clear in-distribution behavior is recency: accuracy on
occurrence 8, which is the final write to the queried key, is `0.6481` versus
`0.2222` for content addressing. Other occurrence indices remain near chance.
On the primary length512/repeat16 OOD setting both arms are at the 32-class
chance level.

## 8. Decision

Reject this occurrence-rotary mechanism at the preregistered gate. It misses
the required `+0.20` OOD gain by a wide margin and does not create
query-addressable older versions. The evidence is more specific than
"position does not help": standard RoPE binding changes which recent value is
easy to read, but the KDA state update still behaves like overwrite memory for
older writes.

Do not implement the learned key-local counter, and do not sweep rotary base,
frequency, beta, seed, width, or training length. A future revisit would need
a qualitatively stronger orthogonal/versioned address-capacity upper bound,
not a tuned version of this failed probe.

## 9. Submission

Not applicable.
