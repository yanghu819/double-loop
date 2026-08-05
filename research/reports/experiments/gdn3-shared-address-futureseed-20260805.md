# P-GDN3-001: Cross-Layer Shared Address Namespace

## 1. Metainfo

- Status: completed; quick falsifier passed and clean scale authorized
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

## 10. CUDA Contract Result

The GPU1-only contract passed on A100 UUID
`GPU-53e9f3b4-2966-65d3-6614-09c540921519` with pinned FLA source
`9c8e42e762fce087c27b673af4922795d9edb85e`.

- zero initialization is an exact identity for output, terminal state, initial
  state gradient, and all pre-existing model gradients;
- the single shared projection receives a finite nonzero first-step gradient;
- all ten recurrent layers consume the same shared address tensor;
- the active graph contains `ChunkGDN2FunctionBackward` and Triton Q/K/V short
  convolutions;
- there is no reverse scan, fallback, search, repair, selector, or task rule.

## 11. Formal Result

Run `gdn3-shared-namespace-s9100-20260805T035329Z-ea6887a` resumed the exact
canonical step9000 FutureSeed checkpoint and trained for the registered 100
steps. The run stayed on GPU1 and clean source SHA
`ea6887aaf08ef1a14ea2cd4f5d9196f45d201f68`.

| Step9100 metric | normal GDN2 | per-layer shared address | GDN3 shared namespace |
|---|---:|---:|---:|
| train CE | 1.9370 | 1.6845 | **1.1634** |
| official 51-55 loop5 blank | 0.2120 | 0.3106 | **0.4816** |
| official 56-60 loop5 blank | 0.2119 | 0.2860 | **0.4361** |
| official 61-64 loop5 blank | 0.1871 | 0.2565 | **0.4260** |
| mean official 51-64 blank | 0.2037 | 0.2844 | **0.4479** |
| train elapsed, 100 steps | 649.9 s | 776.9 s | 694.8 s |
| full-board exact | 0 | 0 | 0 |

Relative to normal GDN2, mean hard blank accuracy improves `+0.2442`, train CE
improves `-0.7736`, and elapsed time increases only `+6.9%`. It also exceeds
the old per-layer shared-address result by `+0.1635` mean hard blank accuracy.
The shared residual learned a substantial but bounded signal: weight RMS
`0.0170`, residual RMS `0.6890`, and relative Q/K changes `0.3056/0.4003`.

## 12. Loop Correction And Visualization

This is not only a better loop1 operating point. Across the 256 fixed boards in
each official hard bucket, mean wrong blank cells evolve as follows:

| Bucket | loop1 | loop2 | loop3 | loop4 | loop5 | improved / same / regressed |
|---|---:|---:|---:|---:|---:|---:|
| 51-55 | 30.027 | 28.406 | 28.234 | 28.148 | 28.109 | 169 / 27 / 60 |
| 56-60 | 33.441 | 32.355 | 32.289 | 32.270 | 32.266 | 140 / 28 / 88 |
| 61-64 | 39.910 | 37.250 | 37.020 | 36.992 | 36.992 | 186 / 28 / 42 |

One mechanically selected 64-blank case changes
`37 -> 31 -> 29 -> 28 -> 27` wrong cells. Another changes
`43 -> 32 -> 29 -> 29 -> 28`. The archived case bank contains input, target,
and loop1-5 predictions for all selected hard failures. No selected case is
solved or almost solved, so this remains a mechanism result rather than a
Sudoku frontier result.

## 13. Decision

P-GDN3-001 passes every registered scale condition: CE, hard blank accuracy,
real recurrent correction, runtime overhead, kernel provenance, and exact
identity at initialization. The result supports the bounded claim that a
single cross-layer address namespace makes FutureSeed-transported GDN2 state
easier for later layers and loops to use.

It does not support a solved-Sudoku or final GDN3 claim because every official
51-64 full-board exact rate is still zero. The only authorized next experiment
is one clean from-scratch D256/L12 full-diversity scale run. Training from
scratch is important: the quick continuation had to rotate a mature stack of
independent layer address bases abruptly, whereas the scale run lets content
addresses, the shared namespace, and FutureSeed co-adapt from initialization.
No address strength, rank, loss, seed, or duration table is authorized.
