# P-GDN3-036: Anchored Dual-Key GDN2

## 1. Metainfo

- Status: approved; implementation
- Date: 2026-08-14
- Benchmark: directional MQAR L1024, four associations
- Seed: 123 only
- Model: D128/L2/H4/K32/V32 pinned-official DPLR GDN2 plus native FutureSeed
- Control: contemporaneous native GDN2+FutureSeed in the same process

## 2. Hypothesis

P-GDN3-031 directly tested the external `GDN_decouple_k` mathematical idea.
Its separately learned erase and write/read keys were trainable and stable, but
their layer cosines collapsed to `0.0483/0.0600`. Erase consequently stopped
targeting the rows populated by write and queried by read; balanced accuracy
collapsed to `0.011`. This rejects unconstrained key independence, not every
form of erase specialization.

P-GDN3-035 independently rejects grouped persistent V-axis lifetime. The next
high-information question is therefore whether erase needs a small
address-local correction while preserving shared row ownership. The candidate
hard-codes that ownership as a geometric invariant rather than hoping training
or a regularizer rediscovers it.

## 3. Configuration

Let `k` be the normalized native write/read key and `r` an independently
predicted normalized raw erase key. The candidate computes

`t = r - ((r*k)/(k*k))*k`,

`c = rho*t`, with fixed `rho=0.5`, and

`k_e = (k+c)/sqrt(1+||c||^2)`.

Because `c` is orthogonal to `k`, every opened erase key has cosine at least
`1/sqrt(1+rho^2)=0.894427` with the write/read key. When the raw erase pipeline
is copied exactly from the write pipeline, `t=0` exactly and the candidate is
the tied DPLR parent function. The derivative through the tangent projection is
nonzero at this initialization, so the new route can learn immediately.

The rest of GDN2 is unchanged: native Q/V/decay/channel erase/write/output,
K32xV32 state, native FutureSeed and one scan. The DPLR update is

`S' = D S - k_e [(D(b*k_e))^T S] + k [(w*v)^T]`.

The candidate adds exactly 33,792 projection/convolution parameters over two
layers, no recurrent state, token, cache, reverse scan or task logic. Both arms
use 10,000 train and 1,000 validation examples, ten epochs, batch32, AdamW
LR1e-3/WD0.1, BF16 and seed123.

## 4. Environment

- Exactly one visible CUDA index0 and its registered UUID
- Pinned FLA source SHA
  `9c8e42e762fce087c27b673af4922795d9edb85e`
- Official `ChunkDPLRDeltaRuleFunctionBackward`, exactly one scan per layer
- Formal source is an exact pushed SHA in a clean detached worktree
- No CPU model smoke or concurrent GPU model/evaluation

## 5. Commands

The committed launcher freezes the exact contract and matched endpoint:

```bash
scripts/run_zoology_anchored_dual_key.sh
```

## 6. Launch Gates And Artifacts

Before training, the CUDA contract must prove all of:

1. exact GPU UUID, pinned FLA/Zoology source and Triton short convolution;
2. exact native parent parameter hash and +33,792 candidate parameters;
3. zero-tangent candidate output is exactly the tied direct-DPLR parent output;
4. tied DPLR agrees with native and explicit GDN2 recurrence within fixed BF16 tolerance;
5. two official DPLR backward paths with finite nonzero and distinct write/erase gradients;
6. exact call mapping, one state, one scan and no fallback;
7. synthetic and production tangent orthogonality and key cosine floor;
8. opened erase-key output/state dependency, head equivariance, sampled spectral norm at most 1.25 and finite terminal state.

Formal artifacts must include control/candidate scores, cases, checkpoints,
comparison JSON, source/config/log hashes and GPU timing/memory.

R1 exited before model construction because the launcher used
`FLA_SOURCE_ROOT` in `PYTHONPATH` without exporting it to the checker process.
The wrapper wrote a non-science contract abort; GPU allocation remained zero.
R2 changes only the missing environment export and preserves the mechanism,
data, training command and all registered gates.

## 7. Registered Result Gates

Activation requires both layers, native FutureSeed, exact initial tie,
correction relative RMS at least `1e-3`, nonzero board/token variation,
projection and convolution movement at least `1e-5`, key cosine minimum at
least `0.89`, tangent orthogonality error at most `5e-3`, sampled transition
spectral norm at most `1.25`, and finite board-varying terminal state.

Quality requires every check:

- balanced accuracy at least 0.35 and at least +0.10 over control;
- future and past accuracy each at least +0.07 over control;
- joint exact at least 0.03 and at least +0.03 over control;
- fewer total errors;
- conditional wrong-key valid-value swap fraction at least 0.05 below control.

Elapsed, post-warm wall and warmed-step ratios must each stay below 1.75x
control, and peak allocation below 1.50x.

## 8. Decision

A full pass authorizes one matched hard-Sudoku transfer. Any contract,
activation, stability, quality or cost miss closes anchored dual-key GDN2 and
the broader decoupled-key line. There is no residual-cap, tangent map,
regularizer, initialization, key-angle, seed, optimizer, width, depth or
duration rescue.

## 9. Submission Record

Not applicable. This is an architecture experiment, not a submission.
