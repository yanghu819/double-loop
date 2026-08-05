# P-GDN3-001: Cross-Layer Shared Address Namespace

## 1. Metainfo

- Status: approved, implementation pending
- Preregistered: 2026-08-05 11:12 CST
- Benchmark: hard 9x9 Sudoku, official 51-64 blank evaluation
- Resource: AIStation task-mode GPU1 only
- Branch: `codex/sudoku-fs-gdn3-mainline`
- Parent evidence: canonical GDN2+native FutureSeed step9000 checkpoint

## 2. Mechanism Question

FutureSeed transfers the terminal recurrent state of one layer into the next.
For GDN2 this state is a KxV matrix. The V axis stores payload, but the K axis is
the address coordinate used by queries, erasure, and writes. Standard GDN2 gives
every layer independent content Q/K projections. A state written in layer `l` can
therefore arrive in layer `l+1` expressed in an address basis that the receiving
layer does not share.

The question is:

> Is cross-layer address incompatibility limiting native FutureSeed, and can one
> shared generic address namespace turn transported state into more usable
> recurrent memory?

## 3. GDN3 Candidate

Construct one stable anchor from token and position embeddings, normalize it
without layer-specific affine parameters, and project it once:

```python
shared_address = W_shared(rms_norm(token_anchor + position_anchor))
q_l = q_content_l + shared_address
k_l = k_content_l + shared_address
```

The exact same `W_shared` output is used in every recurrent layer. Each layer
retains its own content Q/K, V, decay, erase, write, output gate, and channel
mixer. The GDN2 rank-one recurrence, state shape, official FLA chunk function,
and Triton backward remain unchanged.

`W_shared` is initialized to exact zero. At initialization the candidate must be
bitwise identical to the strong GDN2+FutureSeed parent. This creates a clean
optimization path rather than replacing the parent function.

## 4. Why This Is General

This mechanism does not encode rows, columns, boxes, digits, constraints, or a
solver. It addresses a generic compositional memory problem: recurrent state is
transferred between modules that otherwise learn unrelated address coordinates.
The same issue occurs in stacked linear-attention language or retrieval models.

## 5. Prediction

If address compatibility is a real bottleneck, the shared projection should:

- receive a finite nonzero gradient on the first optimizer step;
- reduce hard validation CE relative to the frozen normal GDN2 trajectory;
- raise mean official 51-64 blank accuracy by at least `+0.10` over normal or
  exceed the existing per-layer shared-address result at the same gate;
- improve loop1-to-loop5 wrong-cell reduction, not only loop1 local accuracy;
- stay within 25% training-time overhead and preserve the official FLA kernel.

If it only raises blank accuracy at loop1, leaves later-loop correction flat, or
cannot beat the per-layer address signal, then sharing the basis is not the
missing FutureSeed/GDN3 mechanism.

## 6. GPU Contract Test

Before training, one CUDA-only test must verify:

1. exactly one visible GPU, registered GPU1 UUID;
2. pinned official FLA source and Triton backend;
3. exact zero-init output, terminal-state, initial-state-gradient, and base
   parameter-gradient identity against ordinary GDN2;
4. finite nonzero gradient for `W_shared` on the first backward;
5. all layers consume the same shared address tensor and there is only one
   trainable shared projection;
6. active nonzero `W_shared` changes Q/K and still traverses
   `ChunkGDN2FunctionBackward`;
7. no fallback, reverse scan, selector, repair, search, or task rule.

No CPU model smoke is allowed.

## 7. Formal Falsifier

- Parent: exact canonical GDN2+FutureSeed D192/L10/H6/D32 step9000 train state.
- Candidate-only continuation: step9000 to step9100.
- Data: the same hard/random-order Sudoku curriculum and RNG contract used by
  the address probes.
- Training: loop5, supervision at every loop, same optimizer, scheduler, batch,
  BF16, FutureSeed scale, and official FLA kernel.
- Evaluation: official 51-55, 56-60, 61-64; loop1 through loop5 CE, exact,
  blank accuracy, and wrong-cell trajectories.
- Comparators: frozen normal and already archived per-layer address results.
  They are not rerun.

This is one candidate, one seed, and one fixed duration. There is no tying
strength, rank, LR, loss, width, or checkpoint table.

## 8. Kill and Scale Decision

Kill and archive on provenance mismatch, nonidentity, zero shared gradient,
NaN/OOM, runtime overhead above 25%, or no simultaneous CE/blank/loop signal.
Do not rescue with another initialization or scalar.

Only a passing falsifier authorizes one clean scaling run, using full-diversity
data and at least D224/L12 capacity. That run tests whether the GDN3 mechanism
continues to improve exact closure when data, model/state, and training compute
are increased. A quick-probe win alone is not a paper result.

## 9. Allowed Claim

A successful falsifier supports only:

> Cross-layer address compatibility improves the optimization and recurrent
> correction of FutureSeed-initialized GDN2 on hard Sudoku.

A scalable GDN3 claim requires the subsequent clean full-budget run and must
report full-board exact, not just blank accuracy.
