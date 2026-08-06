# P-FS3-001: Orthogonal Innovation FutureSeed

## 1. Metainfo

- Status: discarded after the registered matched-pair decision
- Date: 2026-08-06
- Source SHA:
  `3e167b6f779065fee50b2b19107c3c5baa7088ee`
- Branch: `codex/fs3-producer-codec-20260807`
- Control/candidate endpoint: exact resume from step3000 to step3100
- Decision time: 2026-08-07 05:01 CST
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

The exact launch commands and environment are frozen in:

- `/huyang2/double-loop/artifacts/launch/p-fs3-001/contract-3e167b6.log`
- `/huyang2/double-loop/artifacts/launch/p-fs3-001/matched-pair-20260806T200859Z-3e167b6.log`
- `/huyang2/double-loop/runs/p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6/config.json`
- `/huyang2/double-loop/runs/p-fs3-001-innovation-s3100-20260806T200859Z-3e167b6/config.json`

Both arms used `--resume_require_exact_state` from:

`/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`

with SHA256
`6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
and parent source
`9f2ee8d1738032bc5f09b55db0b81d507780b376`. Only the candidate
set `future_seed_content_mode=innovation_residual` and
`--resume_allow_future_seed_content_upgrade`.

The original sequence wrapper checked the nonexistent output name
`futureseed_loop_metrics_seed52.json` after the control completed. The actual
registered output is `futureseed_loop_seed52.json`, so the wrapper exited
status1 before launching the candidate. This was an orchestration postcondition
bug, not a model or integrity failure. The candidate was immediately launched
with the exact registered command and unchanged parent/configuration.

## 6. Artifacts

- Implementation: `experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py`
- CUDA contract: `experiments/rwkv_fs_sudoku/check_futureseed_innovation_cuda.py`
- Control run:
  `/huyang2/double-loop/runs/p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`
- Candidate run:
  `/huyang2/double-loop/runs/p-fs3-001-innovation-s3100-20260806T200859Z-3e167b6`
- Matched decision and same-board visualization:
  `/huyang2/double-loop/runs/p-fs3-001-comparison-20260806T210100Z-3e167b6/comparison.json`
  and `comparison.html`
- Decision JSON SHA256:
  `53979f922f5ed7cd9f633ba61074f45eef2188c3ac4a39216d7d843658623b8b`
- Control metric/checkpoint SHA256:
  `7c0bbc1a97485748d2788cf119e744de5c13bf6d6aa3c58faed227a7c4b606b8` /
  `2b9046fa54e4f094eeadee2f88556bc920743ad3e4f9e160d3084d6658fbe294`
- Candidate metric/checkpoint SHA256:
  `88c6ee0b1258527be7c9ef4fa4c30360c277dac862ff2f8d6bcae4da8a4a461f` /
  `35aa4b94e976f91cf53b31d7379fc10589532dcfb043f3500d84046fc3d74c9e`
- Source snapshot SHA256 (both arms):
  `318110ff394c2547e9b4a13a5f17fede7a5bfcf3705f53466aa3374a595ad230`
- Full path/hash manifest:
  `/huyang2/double-loop/runs/p-fs3-001-comparison-20260806T210100Z-3e167b6/manifest.sha256`

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

### 7.1 Matched result

Integrity and activation pass. The contract verified exact zero-init identity,
the pinned FLA/Triton path, active position-QK, finite backward, and the exact
80-parameter delta. Both arms completed with status0 and no
NaN/OOM/fallback/SHA/data/GPU drift. At loop5 the candidate has mean
`abs(tanh(alpha))=0.010348`, innovation fraction `0.866`, and residual
relative RMS `0.010348`.

The quality gates fail:

| Metric | Terminal control | Innovation candidate | Delta |
|---|---:|---:|---:|
| hard 51-64 macro loop5 exact | 0.000651 | 0.000651 | +0.000000 |
| mixed loop5 exact | 0.025391 | 0.025391 | +0.000000 |
| 51-55 loop5 blank | 0.573766 | 0.573623 | -0.000143 |
| 56-60 loop5 blank | 0.503861 | 0.502316 | -0.001544 |
| 61-64 loop5 blank | 0.591923 | 0.591404 | -0.000519 |
| train CE | 0.858617 | 0.857070 | -0.001547 |

Official exact across loops1-5 is identical between arms:

- 51-55: `0/0/0.001953/0.001953/0.001953`
- 56-60: `0/0/0/0/0`
- 61-64: `0/0/0/0/0`

The candidate therefore does not convert its small CE change into an additional
board solve. Same selected-board evidence also rejects the alternate route:

| Range | Shared boards | Control wrong cells loops1-5 | Candidate wrong cells loops1-5 |
|---|---:|---|---|
| 51-55 | 11 | 15.27 / 8.73 / 6.45 / 5.27 / 5.27 | 15.45 / 9.09 / 6.36 / 5.73 / 5.27 |
| 56-60 | 10 | 17.50 / 8.90 / 5.80 / 5.40 / 5.20 | 17.20 / 9.10 / 6.20 / 5.60 / 5.50 |
| 61-64 | 7 | 28.43 / 18.71 / 16.29 / 16.14 / 15.71 | 28.14 / 18.29 / 16.14 / 15.43 / 15.57 |

Only 51-55 has a slightly larger loop1-to-loop5 reduction, while 56-60 and
61-64 do not. That cannot satisfy the all-range same-board clause.

The fresh-process 100-step continuation averages are `15.497` versus
`13.963` effective boards/s: candidate elapsed time is `+10.98%`, just
outside the registered `<10%` cost gate. Peak allocated memory is
`13,186.4 -> 13,875.6 MiB` (`+5.23%`), so memory remains acceptable.
These 100-step averages amortize first-shape compilation but are not a separate
standalone timing benchmark; quality already makes the decision invariant.

## 8. Conclusions

Discard P-FS3-001. The candidate is alive and numerically well behaved, but
emphasizing the component orthogonal to the inherited seed does not improve
hard exact, mixed exact, or broad same-board correction. The negative result is
therefore about the content hypothesis, not dead gradients or a broken kernel.

Do not rescue with alpha initialization, scale, floor, rank, seed, LR, loss,
batch, width, duration, or another analytic residual. The next mechanism must
learn a shared producer compression/update code with materially more expressive
content selection while preserving exact zero-init identity and official
recurrence. It must be preregistered as a new mechanism question, not a
continuation of this scalar gate.

## 9. Submission Record

Not applicable.
