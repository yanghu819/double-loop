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

The fixed run completed with status 0 on CUDA index 0, UUID
`GPU-d2877fe4-641c-fe64-2a74-8abca47c292f`. All source, data, initialization,
pinned-FLA and Zoology provenance checks passed.

At L1024, the historical-order causal arm reached balanced/future/past/joint
accuracy `0.015/0.009/0.021/0`. The following native FutureSeed arm reached
only `0.10825/0.0990/0.1175/0`. Its validation curve remained near chance
through epoch 3 and rose only to `0.02075/0.03775/0.08375` at epochs 4/6/7,
ending at `0.10825`. It did not reproduce the historical epoch-4-to-6 jump
(`0.1965->0.71575`) or endpoint `0.7475`.

The same process reproduced the L64 sanity carrier: native FutureSeed reached
balanced/future/past/joint `0.99525/0.9920/0.9985/0.981`. This rules out a
broken model import, dataset or FutureSeed path and localizes the discrepancy
to the unstable long-sequence optimization trajectory.

The L1024 FutureSeed warmed throughput was `1,349,294.7` tokens/s with
`1,084,782,080` peak benchmark bytes; training plus validation took
`101.447` seconds after warmup. No NaN, OOM or fallback occurred.

## 8. Decision

The preregistered `<=0.40` branch fires. Retire historical L1024 balanced
accuracy `0.7475` as a non-reproducible absolute architecture gate on the
current stack. Restoring exact historical source and exact causal-to-FS
single-process order is insufficient, so process order is not the missing
explanation.

Future L1024 architecture decisions must use a contemporaneous control in the
same source, process, GPU task and arm order, plus a relative improvement gate.
Historical scores remain useful descriptive evidence but cannot pass or fail a
new candidate. No additional carrier seed, LR, epoch or subprocess-isolation
run is authorized.

## 9. Provenance

- Historical source: `77e5539fc0ef74231cab658bd610b24254fdcfa7`.
- Historical run:
  `/huyang2/double-loop/runs/zoology-gdn2-fs-length1024-20260804T095300Z-77e5539`.
- Current native reference:
  `/huyang2/double-loop/runs/p-fs2-007-surprise-replay-l1024-r2-20260813T075300Z-dc1dd53`.
- Reproduction run:
  `/huyang2/double-loop/runs/p-diag-carrier-002-historical-order-20260814T063454Z-77e5539`.
- Score SHA256:
  `f26dd46bef652f8f88136fcd11a2a52b4dcd1f86b07563f8750424263031e255`.
- Contract SHA256:
  `9d7f7a664597ec1233386caad1a0f542b978bfc3732055dbc5fd89b53ddb7b0d`.
- Formal log SHA256:
  `8d5b069f1bc52ae51b5e64ca48e90346ac81229f5228923f275002d9f34df905`.
