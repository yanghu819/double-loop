# P-GDN3-066 Official Mesa Least-Squares FutureSeed

## 1. Metainfo

- Plan: `P-GDN3-066`
- State: complete; discarded
- Decision field: directional MQAR L1024/K4, not Sudoku
- Resource: one AIStation A800 80GB, CUDA index 0 only
- Frozen reference: `P-GDN3-059` Momentum DeltaNet + native `[S,M]` FutureSeed

## 2. Hypothesis

P059's remaining errors are predominantly correct-value/wrong-key swaps. All
151 swaps use an adjacent write rank and 150/151 preserve direction. More
slots, value width, write controllers, dual keys, side memories, phase mixing,
and pre/post-scan residuals have failed. The missing operation is therefore not
more storage but an address-interference-aware read.

Mesa stores decayed key Gram and key-value sufficient statistics and reads by
solving a regularized normal equation. This should separate correlated keys
without a Sudoku selector or another learned wrapper. Native FutureSeed carries
the complete `[Hkk,Hkv]` state to the receiver with one positive joint scale,
preserving their relative units.

## 3. Fixed Configuration

- Arm: `future_seed_mesa`
- Model: D128/L2/H4/K32/V32, output gate enabled
- Recurrence: exact pinned official FLA `chunk_mesa_net`
- State per layer: `Hkk[H,32,32] + Hkv[H,32,32] = 8,192` values
- Solver: 30 conjugate-gradient iterations, fixed official default
- Regularizer lower bound: `0.25`, fixed official default
- FutureSeed: one adjacent-layer route, joint RMS normalization of `[Hkk,Hkv]`
- Data: fixed directional MQAR L1024/K4, 10,000 train and 1,000 test examples
- Train: 10 epochs, batch32, seed123, LR/WD inherited unchanged
- Initialization: matched compatible shell tensors from the frozen parent;
  Mesa recurrence parameters train from scratch because this is a foundational
  recurrence, not a 100-step zero-init graft
- Sweep: none

## 4. Environment and Provenance

- FLA marker: `9c8e42e762fce087c27b673af4922795d9edb85e`
- Official source: `fla/layers/mesa_net.py` and `fla/ops/mesa_net/*`
- License: MIT; no external source is copied into this repository
- Runtime: pinned PyTorch 2.8 / Triton >=3.4 environment
- Launch requires exact pushed SHA, GitHub readback, clean detached worktree,
  exact parent/frozen-artifact hashes, target GPU UUID, and no other compute app

## 5. Contract and Commands

The strict contract must prove:

1. Exact pinned source paths and file hashes.
2. Exactly two `ChunkMesaNetFunctionBackward` paths and no fallback.
3. Finite nonzero gradients for Q/K/V, decay, write, lambda, output, output
   gate, both initial states, producer terminal state, and the receiver
   FutureSeed gate.
4. Official chunk output/state parity against `naive_mesa_net_exact` with a
   nonzero initial state: output relative RMS <=0.05 and both state relative
   RMS <=0.01.
5. Head permutation equivariance <=0.005 relative RMS.
6. Exact parameter/state accounting, symmetric PSD Hkk up to 0.01 numerical
   tolerance, finite board variation, and endpoint CG relative error <=0.05.

Launch command:

```bash
setsid env EXPECTED_SOURCE_SHA=<pushed-sha> \
  bash scripts/run_zoology_mesa_futureseed.sh
```

## 6. Artifacts

R1 source `3bc35d08` stopped before any model/CUDA execution. The provenance
checker called `inspect.getfile` on the `@torch.compiler.disable`-decorated
`chunk_mesa_net` function and therefore saw TorchDynamo's wrapper file instead
of the operator module. The actual `ChunkMesaNetFunction`, layer path, module
path, and all pinned file hashes were unchanged. R2 validates the module
`__file__` plus operator class path and exact hashes. This is a non-science
checker correction; mechanism, data, initialization, budget, and gates are
unchanged.

R2 source `f0181c77` completed the expensive official operator execution and
proved material incoming-state use in the receiver output (relative RMS
`0.126012`, finite), but the checker also required the old initial state to
remain visible in the terminal state after the full sequence. Its measured
terminal difference was exactly zero. That extra condition was not a registered
contract item and contradicts the intended stable forgetting behavior: the seed
must affect receiver computation, while fresh terminal sufficient statistics
are separately required to be nonzero, board-varying, symmetric/PSD, and
differentiable. R3 therefore keeps output dependence as a hard gate and records
terminal carry without requiring stale initial-state retention. This is a
non-science checker correction; recurrence, data, initialization, budget, and
all quality/cost gates remain unchanged.

R3 exact pushed/read-back source `d1cf70bd4f86612f338373fee639136162298bd0`
ran from the clean detached worktree
`/huyang2/double-loop/worktrees/p-gdn3-066-r3-d1cf70b` as
`p-gdn3-066-official-mesa-fs-r3-20260816T093556Z-d1cf70b`.
The strict contract passed: both layers used official
`ChunkMesaNetFunctionBackward`, all registered recurrence, initial-state and
FutureSeed gradients were finite and nonzero, incoming-state output dependence
was `.126012` relative RMS, head permutation errors were zero, and official
chunk/reference parity was `.002488/.000327/.000258` for output/Hkk/Hkv.
Contract JSON SHA256 is
`74353ddf95ab8d2ae89462f438f1721203faf1f5545523cfaafee48645ad879b`.

The fixed endpoint completed naturally and closed the configuration. P059
versus Mesa balanced/future/past/joint accuracy is
`.94425/.95150/.93700/.82400 -> .00975/.00650/.01300/0`. Total errors rise
`223->3961`. Wrong-key swaps are `107`, but this is not an ownership gain:
3,645 native-correct queries become other wrong values, 96 become wrong-key
errors, and only three prior errors repair. Hkk remains symmetric/PSD and the
CG residual is small (`.00191/.00218`), yet endpoint effective rank is only
`1.079/1.169` out of 32. The global sufficient statistic therefore collapses
to an almost one-dimensional address geometry instead of preserving the
directional owner coordinates required by MQAR.

The endpoint records `requires_grad=false` only because diagnostics run under
evaluation/no-grad. It does not override the strict training contract, which
proved the producer terminal state and receiver FutureSeed path are connected.
The active FutureSeed gate is `.49925` and the receiver joint seed RMS is
`.10702`; quality failure is not a dead path. Candidate/control elapsed,
post-warm, warmed-step and allocation ratios are
`.8235/.8339/.8828/.7963`, so all cost gates pass. Full-run telemetry has 77
samples: overall/active mean utilization `16.68%/47.56%`, peak `83%`, peak
observed memory `1,816 MiB`; the overall mean includes source snapshot,
reference validation and CPU-side endpoint analysis.

Comparison/checkpoint/source-snapshot SHA256 values are respectively
`de7d012fe7c72dd92ebd1cfc38903f70e459e62983939624f160e09188ae8ca2`,
`032efb86c71bcfe76831e70aec56180525998cf6c52888aec3bc04245cc453ec`,
and `6ea9fa59193216062399227cf5bf903e53947f3805ebad2e5cdd2e467e77d9cd`.

## 7. Registered Decision Gates

Quality must beat P059 rather than merely recover old GDN2:

- balanced accuracy >= P059 +0.01 (`>=0.95425`)
- future and past accuracy regress by at most 0.005
- joint exact >= P059 +0.01 (`>=0.834`)
- total errors <=180
- wrong-key valid-value swaps <=105
- wrong-key swap share <=0.60713

Cost and stability:

- elapsed, post-warm wall, and warmed-step ratios each <=3.0x P059
- peak allocation <=2.0x P059
- two active Mesa layers, exactly one active FutureSeed route
- finite nonzero state/board variation, stable Hkk geometry, CG error <=0.05

Any integrity, activation, stability, quality, or cost miss closes this exact
Mesa configuration. No CG/lambda/output-gate/FutureSeed-gate/seed/LR/loss/
batch/width/depth/duration rescue is authorized.

## 8. Conclusions

Discard P-GDN3-066. Official Mesa is fully active, stable, faster than P059,
and correctly connected through native joint `[Hkk,Hkv]` FutureSeed, but it
does not learn directional L1024 binding. Its low-rank Gram geometry turns the
sequence into a global regression problem and loses token ownership. This is
strong evidence against least-squares sufficient-statistic memory for the
current regime, not evidence against linear memory generally.

Close CG count, lambda, output gate, seed gate, normalization, seed, LR, loss,
batch, width, depth and duration rescue. The next GDN3 candidate must change
the live recurrent edit while preserving local owner information; another
global statistic, readout wrapper or capacity increase is not authorized.
There is no better GDN3 or FS2 claim from P066.

## 9. Submission Record

Not applicable.
