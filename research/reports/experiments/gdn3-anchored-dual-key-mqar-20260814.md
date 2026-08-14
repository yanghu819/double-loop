# P-GDN3-036: Anchored Dual-Key GDN2

## 1. Metainfo

- Status: discarded; completed_rejected
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

R2 completed from exact clean pushed source
`4b01640c7ff6b5c159fe8f61d5b37c5cdb958ab6`. The strict contract passed:
zero-tangent output identity was exact, two official
`ChunkDPLRDeltaRuleFunctionBackward` paths were present, both new branches had
finite distinct gradients, the synthetic key cosine minimum was `0.89442724`,
and the opened erase route changed output/state while preserving head
equivariance and bounded transition geometry.

The formal endpoint rejects the mechanism. Contemporaneous control versus
candidate balanced/future/past/joint accuracy was
`0.36025/0.3675/0.3530/0.014` versus
`0.01375/0.0145/0.0130/0`. Errors rose `2559->3945`, and future/past CE rose
from `1.9276/2.0239` to `4.5671/4.5668`. The conditional wrong-key swap
fraction fell `0.552950->0.039544` only because useful retrieval collapsed.

This is not an inactive or unstable-module failure. Both layer corrections
reached relative RMS `0.459802/0.463753`; key cosine mean stayed
`0.908641/0.907714`, key cosine minimum stayed `0.893630/0.893207`, and
terminal state remained finite. Elapsed/post-warm/warmed/allocation ratios
were `0.94294/0.95390/1.49283/1.35614`, all inside the registered budget.
Quality alone closes the experiment.

P031 and P036 jointly show that GDN2 does not merely require erase and write
keys to share a broad address neighborhood. Unconstrained separation destroys
ownership, and a strongly bounded erase-only tangent correction still breaks
the read/write/erase closure. Close the complete decoupled-key family without
rho, tangent map, regularizer, initialization, angle, seed, optimizer, width,
depth or duration rescue. No Sudoku transfer is authorized.

Artifacts:

- run: `/huyang2/double-loop/runs/p-gdn3-036-anchored-dual-key-r2-l1024-20260814T125026Z-4b01640`;
- contract SHA256: `48604e3bbd4b1640ea5362571e5c9c12d0c0607a97b138eca9fa2ce224ce116e`;
- comparison SHA256: `0b4a7d61af366ee131afadff27609d837a30669f6eec9c4aa316916647c9036b`;
- control/candidate score SHA256: `d47db565cf42484667abf9dcfc0a7d1071ae3ec44dc83ef3728f23ceb505c2fd` / `1bae339a17a15518f3124a21cc6408536dcd8f092362f6f429f396908e5fbcba`;
- candidate checkpoint SHA256: `87c19c501047fbc22823ba02307d0025039311dc46237e9c25bd4d4aaaa6c6dc`;
- artifact manifest SHA256: `2aabc41f4107cf6ee184095a910c3dc02ba9923b1757fdcf9105343dbcb99575`.

## 9. Submission Record

Not applicable. This is an architecture experiment, not a submission.
