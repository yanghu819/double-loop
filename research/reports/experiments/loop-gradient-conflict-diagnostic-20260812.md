# P-LOOP-001: Per-Loop FutureSeed/GDN2 Gradient Conflict Audit

## 1. Metainfo

- Status: completed; FutureSeed gate branch admitted, GDN2 groups rejected
- Parent: position-QK GDN2 plus native terminal FutureSeed, exact step3000
- Task: official/full-diversity hard 9x9 Sudoku, 51-64 blanks
- Intervention: none; this diagnostic does not alter logits or parameters

## 2. Hypothesis

The remaining exact-board cliff may be an optimization conflict rather than a
missing state wrapper. On the same hard boards, early loops may push native
FutureSeed edges or the GDN2 address/edit parameters in directions opposed to
the gradients needed by late corrective loops. Equal all-loop CE would then
cancel useful closure gradients even though loop1-to-loop5 correction is real.

This is not another delayed loss, learned loop gate, state router, cache,
preconditioner or memory-capacity arm. Those nearby families are already
closed. The audit asks whether a gradient-coordination intervention has a
measurable target before any candidate training is allowed.

## 3. Configuration

Load the exact SHA-locked D256/L12/H8/K32/V32 step3000 checkpoint. Use three
fixed official test batches of eight boards from 51-55, 56-60 and 61-64 blanks,
five loops, BF16, blank weight 8, canonical evaluation order, and the unchanged
pinned-official FLA GDN2/Triton graph.

For every loop CE, compute `torch.autograd.grad` without mutating `.grad` or
updating an optimizer. Partition all trainable parameters into:

1. native `future_seed_logit` edges;
2. GDN2 Q/K and Q/K ShortConv address parameters;
3. all remaining official GDN2 edit/readout parameters;
4. the shared embedding/channel/norm/output shell.

Report per-loop norms, all pairwise cosine matrices, early `(L1+L2)` versus
late `(L4+L5)` cosine, adjacent and all-pair conflict fractions, and the
cancellation ratio `||sum g_l|| / sum ||g_l||`. Also report loop exact, blank
accuracy and wrong blank cells. The shell is diagnostic context and cannot
admit a FutureSeed/GDN candidate.

## 4. Environment And Integrity

The run must use one visible AIStation GPU as CUDA index 0, a clean detached
worktree from a pushed SHA, parent checkpoint SHA256
`6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`,
parent source `9f2ee8d1738032bc5f09b55db0b81d507780b376`, pinned FLA source
`9c8e42e762fce087c27b673af4922795d9edb85e`, exactly 12 official GDN2
layers and `ChunkGDN2FunctionBackward`. CPU model fallback and concurrent GPU
model/eval are forbidden.

## 5. Commands

The pushed detached worktree runs:

```bash
EXPECTED_UUID=<admitted-uuid> scripts/run_loop_gradient_conflict_diagnostic.sh
```

## 6. Artifacts

The launcher writes `loop_gradient_conflict.json`, `run.log`, `gpu.txt`, source
provenance and SHA256 manifests below one immutable
`/huyang2/double-loop/runs/p-loop-001-*` directory. No checkpoint is created.

R1 `p-loop-001-gradient-conflict-20260812T135105Z-8d933dc` exited before
gradient measurement because the diagnostic retained only Python object IDs
while traversing the large five-loop autograd graph. Object-ID reuse caused an
incorrect `ChunkGDN2FunctionBackward count=0` provenance verdict. The run wrote
`abort.json` with `scientific_failure=false`; it did not update parameters or
produce a science result. R2 retains autograd node objects during traversal and
changes no registered scientific condition or threshold.

R2 `p-loop-001-gradient-conflict-r2-20260812T140000Z-f99b4a7`
completed with `status=0` on CUDA index 0, A100-SXM4-80GB UUID
`GPU-0da20a4f-5e67-e47d-7aab-8c6efa2864ad`. It retained 180 official
`ChunkGDN2FunctionBackward` nodes per hard range, used 3,502.73 MiB peak
allocation, and completed in 340.05 seconds. The result JSON SHA256 is
`528adc5a77740d73b59d89780651449c65a50bbb2eaaef680a601455b73aef19`;
the clean detached source was exact pushed SHA `f99b4a7b007092a2d6289b4e619d594319232547`.

## 7. Frozen Admission Gate

A parameter group admits exactly one later gradient-coordination candidate only
when the same group qualifies on at least two of the three hard ranges, and
each qualifying range also has loop5-minus-loop1 blank accuracy at least
`+0.02` with fewer wrong blank cells.

Group qualification requires all of:

- early-vs-late cosine `<=-0.10` for FutureSeed gates or `<=-0.05` for either
  GDN2 group;
- all-pair conflict fraction `>=0.30`;
- cancellation ratio `<=0.75`;
- finite nonzero gradients and the official backward path.

If no group qualifies on two ranges, close loop-gradient coordination and run
no 100-step candidate. If one or more groups qualify, preregister one candidate
limited to the strongest qualifying group, with a contemporaneous runtime
control and the existing hard-exact/mixed-exact quality gates. The candidate
mechanism and cost ceiling must be committed before it is launched.

## 8. Conclusions

The frozen gate admits only `future_seed_gate`:

| range | loop1->5 blank | wrong cells | FS early/late cosine | pair conflict | cancellation | FS admitted |
|---|---:|---:|---:|---:|---:|---|
| 51-55 | 0.541284 -> 0.587156 | 25.00 -> 22.50 | -0.904763 | 0.40 | 0.529537 | yes |
| 56-60 | 0.520971 -> 0.538631 | 27.125 -> 26.125 | +0.996918 | 0.00 | 0.996945 | no |
| 61-64 | 0.519531 -> 0.585938 | 30.75 -> 26.50 | -0.718710 | 0.40 | 0.698487 | yes |

The GDN2 address and edit groups have zero negative pairs on all three ranges;
their early/late cosine is respectively `0.916..0.970` and `0.873..0.888` on
56-64. The shared shell is also strongly aligned. This rejects a global
optimizer-conflict explanation and does not authorize another GDN state,
address, or edit wrapper.

The matrix identifies opening credit more narrowly: in the two admitted
ranges, loop1 versus loops3-5 has cosine about `-0.94/-0.86`, while loops3-5
are `0.998..1.000` aligned. Therefore exactly one forward-invariant FutureSeed
gate-gradient candidate is admitted. It must protect the continuation gradient
from a conflicting loop1 component while preserving the original total gate
gradient norm; no loss reweight, per-head variant, scale, seed, LR, batch or
duration sweep is authorized. This diagnostic is causal admission evidence,
not itself a quality score.

The admitted successor is registered as `P-LOOP-002` in
`research/reports/experiments/futureseed-opening-projection-20260812.md`. It
recovers the mean loops2-5 continuation gradient from loop1 and the canonical
equal-five-loop gradient, then performs a training-only FS-gate projection;
it does not reopen a GDN2 mechanism family.

## 9. Submission Record

Not applicable.
