# P-GDN3-037: Committed-Residual Target GDN2

## 1. Metainfo

- Status: approved; implementation
- Date: 2026-08-14
- Benchmark: directional MQAR L1024, four associations
- Seed: 123 only
- Model: D128/L2/H4/K32/V32 plus native FutureSeed
- Control: contemporaneous pinned-official GDN2 in the same process

## 2. Hypothesis

P031 and P036 close the decoupled-key neighborhood. A useful GDN2 edit needs
one coherent address for read, write and erase; even a bounded erase-only
tangent displacement with mean key cosine near `0.91` destroys retrieval.
P021, P025 and P029 further show that an extra write, a separate correction
bank, or a product of edits does not repair binding.

The remaining update-level question is whether GDN2's coordinate-wise erase
gate and independently gated write target prevent one token from committing a
single coherent prediction error. Native GDN2 updates

`S' = D S - k [(b*k)^T D S] + k (w*v)^T`.

The erase response is weighted by a K-coordinate vector `b`, whereas the write
target is weighted independently along V by `w`. The candidate keeps every
projection and the exact coherent key, but converts the edit into one bounded
closed-loop residual:

`u = w*v`, `beta = mean_K(sigmoid(b_logits))`,

`S' = D S + beta*k*(u - k^T D S)^T`.

If wrong-key failures are amplified by an incoherent committed edit rather
than missing state capacity, this exact residual correction should improve
directional binding without another key, state, scan, cache or controller.

## 3. Configuration

Both arms use the same D128/L2/H4/K32/V32 GDN2 Q/K/V/f/b/w/g/o projections,
Triton short convolutions, K32xV32 state and native FutureSeed. Candidate and
control initial tensors must be byte-identical and parameter counts equal.
Only the recurrent equation changes. The candidate maps the equation to one
pinned-official DPLR call:

- `alpha = exp(g)*k`;
- `beta_dplr = -beta*k`;
- `value_dplr = beta*(w*v)`;
- query and additive key are the same normalized `q` and `k` as GDN2.

Both arms use 10,000 train and 1,000 validation examples, ten epochs, batch32,
AdamW LR1e-3/WD0.1, BF16 and seed123. This is a from-scratch foundational
recurrence comparison, not a zero-init Sudoku graft.

## 4. Why Existing Failures Do Not Cover It

- P005 only perturbed erase/write gate statistics for 100 steps on a mature
  Sudoku checkpoint; it did not replace the committed equation or train the
  recurrence from scratch.
- P021 and P029 add edits; P037 uses exactly one edit and one state.
- P030/P031/P036 split erase and write addresses; P037 uses exactly one key.
- P025 stores residuals in an independent bank; P037 commits the residual
  directly into the native state.
- Stock GatedDeltaNet uses a scalar target and a different projection family;
  P037 retains GDN2's vector K decay and V-wise target gate.

## 5. Environment And Contract

- Exactly one visible CUDA index0 and its registered UUID
- Pinned FLA source SHA
  `9c8e42e762fce087c27b673af4922795d9edb85e`
- Official `ChunkDPLRDeltaRuleFunctionBackward`, one scan per layer
- Formal source is an exact pushed SHA in a clean detached worktree
- No CPU model smoke or concurrent GPU model/evaluation

The strict CUDA contract must prove:

1. exact GPU, source, data and Triton provenance;
2. byte-identical control/candidate initialization and equal parameter/state
   counts;
3. exact call mapping to the registered residual equation;
4. agreement between official DPLR and an explicit FP32 recurrence within the
   fixed BF16 tolerance, including nonzero incoming state;
5. two official backward paths and finite nonzero q/k/v/f/b/w gradients;
6. beta in `[0,1]`, one coherent key, one state and one scan;
7. bounded sampled transition, finite board-varying terminal state and no
   fallback.

## 6. Registered Prediction And Gates

Activation requires both layers, native FutureSeed, beta and target variation,
committed-residual relative RMS at least `1e-3`, nonzero b/w gradients, sampled
transition spectral norm at most `1.05`, and finite board-varying terminal
state in `[1e-4,1e4]` RMS.

Quality requires every check:

- balanced accuracy at least `0.35` and at least `+0.10` over control;
- future and past accuracy each at least `+0.07` over control;
- joint exact at least `0.03` and at least `+0.03` over control;
- fewer total errors;
- conditional wrong-key valid-value swaps at least `0.05` below control.

Elapsed, post-warm wall and warmed-step ratios must each be below `1.50x`
control, and peak allocation below `1.25x`.

## 7. Kill Criteria

Any integrity, activation, stability, quality or cost miss closes the exact
committed-residual target equation. There is no beta reducer, residual scale,
target gate, decay, kernel, initialization, seed, LR, loss, batch, width,
depth or duration rescue. Only a full pass authorizes one matched hard-Sudoku
transfer.

## 8. Required Artifacts

Archive control/candidate metrics and cases, wrong-key taxonomy, validation
curves, checkpoints, exact equation diagnostics, state geometry, throughput,
memory, source/config/log/checkpoint hashes and GitHub provenance.

## 9. Submission Record

Not applicable. This is an architecture experiment, not a submission.
