# P-FS3-001: Orthogonal Innovation FutureSeed

## 1. Metainfo

- Status: approved successor; implementation prepared, no GPU run launched
- Date: 2026-08-06
- Benchmark: official/full-diversity hard 9x9 Sudoku
- Compute: AIStation task-mode GPU1 only
- Seed: 52 only
- Parent mechanism: pinned official-FLA GDN2 plus native FutureSeed

## 2. Hypothesis

Native FutureSeed passes a producer layer's terminal recurrent state to the next
layer after per-example/head RMS normalization and a learned head gate. A
producer terminal state contains both inherited seed direction and evidence
newly written by that layer. Unit normalization can let the inherited component
dominate even when the newly written component carries the useful refinement.

The candidate isolates the producer's state innovation orthogonal to its input
seed and adds a bounded, zero-initialized per-layer/head amount of that
innovation to the unchanged terminal state. If late hard-Sudoku closure is
limited by stale seed direction, this should improve loop3-5 correction without
changing the state basis, transfer radius, recurrent kernel, or first transfer.

## 3. Configuration

For receiver layer `l >= 2`, let `T` be producer layer `l-1`'s terminal state and
`I` the exact initial state consumed by that producer. Per example and head:

```text
R = T - <T,I> / max(<I,I>, eps) * I
R_bounded = R / max(rms(R), 0.1 * rms(T)) * rms(T)
seed_l = T + tanh(alpha_lh) * R_bounded
```

`alpha_lh` has shape `(layers-2, heads)` and initializes to exactly zero. P004's
D256/L12/H8 shape therefore adds 80 no-weight-decay scalars. The first
layer0-to-layer1 transfer remains canonical because layer0 has no imported seed.
The existing unit RMS normalization and learned head gate still run after this
composition. The residual RMS is bounded by terminal-state RMS and its direction
is orthogonal to the producer input seed before BF16 casting.

The mechanism is valid only with GDN2, layer-scope FutureSeed, fixed zero-decay
updates, unit normalization, canonical head gates, and no multihop readout. It
does not add a scan, recurrent step, layer, task rule, search, repair, selector,
oracle state, or Sudoku feature.

The decision experiment will be one matched 100-step continuation pair from the
same strongest frozen Sudoku checkpoint available after P-GDN3-004 resolves.
Both arms preserve checkpoint optimizer/RNG/data order and all existing model
settings. The control uses `future_seed_content_mode=terminal`; the candidate
uses `innovation_residual`. No second seed or duration/LR/loss/width table is
allowed.

The candidate resume must explicitly set
`--resume_allow_future_seed_content_upgrade` together with exact-state resume.
The loader accepts only a terminal-to-innovation contract change, exactly one
missing tensor (`reasoner.future_seed_innovation_scale`), no unexpected tensor,
and a deterministic optimizer-group expansion. Any broader migration fails
closed. The control never sets this authorization flag.

## 4. Environment

- SSH alias: `aistation-task-gpu1`
- Required visible CUDA index: `0`
- Required UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Persistent root: `/huyang2/double-loop`
- Source policy: pushed SHA in a clean detached worktree
- Kernel policy: pinned official FLA GDN2 chunk/Triton, fail closed
- CPU model smoke and GPU2: forbidden

## 5. Commands

Commands and immutable source/checkpoint hashes remain intentionally blank
until P-GDN3-004 resolves. Before either continuation arm, run exactly one CUDA
contract using `check_futureseed_innovation_cuda.py`. It must verify:

- one visible target UUID;
- exact zero-init model-output and every-layer terminal-state identity;
- parameter delta exactly `(L-2)*H`;
- nonzero finite innovation-scale gradient;
- bounded orthogonal residual formula;
- active position-QK path when composed with P004;
- `ChunkGDN2FunctionBackward`, pinned FLA/Triton, and no fallback.

No model process starts while P-GDN3-004 is healthy.

## 6. Artifacts

- Implementation: `experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py`
- CUDA contract: `experiments/rwkv_fs_sudoku/check_futureseed_innovation_cuda.py`
- Run config, logs, score JSON, checkpoints, source snapshot, and loop
  visualization: pending the registered launch

## 7. Predictions And Gates

The matched continuation pair is allowed only if the selected frozen parent has
nonzero hard exact and genuine loop correction. Otherwise the parent is not a
valid carrier for this FutureSeed content test.

- Integrity: all CUDA/source/data/checkpoint/metric contracts above pass.
- Activation: mean `abs(tanh(alpha)) >= 1e-4`, finite nonzero innovation
  fraction, and nonzero residual relative RMS by step100.
- Primary quality: hard 51-64 macro loop5 exact improves by at least `+0.02`
  versus the matched 100-step terminal control, with no individual official
  blank range regressing by more than `0.01`.
- Alternate quality: mixed loop5 exact improves by at least `+0.03`, 61-64
  exact is non-regressive, and loop3-5 wrong-cell correction improves on the
  same boards.
- Cost: warmed step time overhead below 10 percent and no material peak-memory
  regression beyond allocator noise.

Kill and discard on any integrity failure, dead mechanism, quality miss, broad
blank-only gain without stronger exact/loop correction, NaN/OOM/fallback, or
cost miss. Do not rescue with innovation scale initialization, floor, seed, LR,
loss, batch, width, duration, or another projection variant.

## 8. Conclusions

Pending. A pass would support a specific FutureSeed3 claim: separating newly
written recurrent evidence from inherited state direction improves cross-layer
future-information transfer while preserving the native state coordinates. A
miss closes orthogonal innovation emphasis and redirects to a learned producer
compression formulation, not another scalar/content gate or basis adapter.

## 9. Submission Record

Not applicable.
