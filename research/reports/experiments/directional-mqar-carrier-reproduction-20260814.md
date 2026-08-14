# P-DIAG-CARRIER-002: Exact Historical Carrier Reproduction

## 1. Question

Is the historical native K32 GDN2 plus FutureSeed directional-MQAR L1024
result (`0.7475` balanced accuracy) reproducible on the current A100 when the
exact source and original single-process arm order are restored?

## 2. Motivation

The historical score and the current P-FS2-007 native control use identical
model configs, parameter initialization hash
`3b0c133410eaed1135224cc0acb094705655cc7e97ea6beeba17b40beab1a368`,
train/test hashes, pinned FLA SHA `9c8e42e`, pinned Zoology SHA `1ad20d1`, and
the same A100-SXM4-80GB architecture. Nevertheless, balanced accuracy is
`0.7475` historically and `0.30625` currently, with divergence visible in
epoch0.

The material protocol difference is process order. Historical source
`77e5539` runs causal GDN2 first and FutureSeed GDN2 second. The current native
control runs first in its process. Triton
compilation/autotuning and CUDA execution order can therefore alter the
optimization trajectory despite reset Python/PyTorch seeds. Continuing to
judge mechanisms against an unresolved baseline would confound architecture
with runtime protocol.

## 3. Intervention

Run exact pushed source `77e5539fc0ef74231cab658bd610b24254fdcfa7`
unchanged in a clean detached worktree. Invoke its original
`experiments.zoology_mqar.length_scaling` entrypoint at L1024 so the exact arm
order is preserved:

```text
causal_gdn2 -> future_seed_gdn2
```

Use D128/L2/H4/K32/V32, four pairs, 10,000/1,000 examples, batch32, ten
epochs, seed123, the locked data cache, and the pinned official FLA/Triton
runtime. There is no model change, new parameter, new state, selector, search,
or Sudoku logic.

## 4. Falsifiable Prediction

- If FutureSeed balanced accuracy is at least `0.70` and its curve shows the
  historical late rise by epochs4-6, the historical ordered carrier is
  reproducible. Future architecture runs must use that same process protocol;
  isolated first-arm scores are a different runtime condition.
- If balanced accuracy is at most `0.40`, the historical score is not
  reproducible on the current runtime. Retire `0.7475` as an absolute gate and
  freeze the contemporaneous isolated carrier instead.
- A result in `(0.40, 0.70)` is ambiguous. Do not launch another architecture
  on L1024 until each arm is isolated in a deterministic fresh subprocess.

No branch authorizes a seed, LR, loss, batch, width, depth, epoch or duration
sweep.

## 5. Integrity Contract

Require one visible A100 CUDA index0, the registered UUID, pinned FLA source
and wheel hashes, official GDN2/Triton provenance, exact Zoology SHA, exact
historical source SHA, identical train/test hashes and the exact historical
FutureSeed initialization hash. There must be one GPU compute process and no
fallback, NaN or OOM.

## 6. Budget And Kill Rule

This is one exact two-arm reproduction, expected to take under ten minutes.
Any provenance or integrity mismatch aborts the process. A completed result is
accepted as a carrier decision regardless of score and is never rescued.

## 7. Results

Pending.

## 8. Decision

Pending the one fixed run.

## 9. Provenance

- Historical source: `77e5539fc0ef74231cab658bd610b24254fdcfa7`.
- Historical run:
  `/huyang2/double-loop/runs/zoology-gdn2-fs-length1024-20260804T095300Z-77e5539`.
- Current native reference:
  `/huyang2/double-loop/runs/p-fs2-007-surprise-replay-l1024-r2-20260813T075300Z-dc1dd53`.
