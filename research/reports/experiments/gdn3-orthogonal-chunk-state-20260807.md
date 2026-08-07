# P-GDN3-009: Orthogonal Chunk-State Transport

## 1. Metainfo

- Status: discarded at strict CUDA contract; no probe or formal continuation
- Date: 2026-08-07
- Branch: `codex/gdn3-orthogonal-chunk-state-20260807`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent: D256/L12/H8/K32/V32 position-QK GDN3 plus native terminal FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Evidence Boundary

P-GDN3-005 couples erase and write gates but does not mix live state rows.
P-GDN3-006 adds an independent residual state expert. P-GDN3-007 reads the
incoming layer state once before the token scan. P-GDN3-008 performs an
unconstrained second sweep after the complete scan; it improves every hard
blank range but does not close boards and drives terminal residual relative RMS
to `4.7109e8` by loop5.

The untested boundary is stable address transport *inside* the live recurrent
scan. Official GDN2 has channel-wise decay and rank-one erase/write updates but
no norm-preserving mechanism that moves already stored information between K
address rows. P009 tests that operation at the official kernel's native
64-token chunk boundary. It is not another full sweep, pre-scan residual,
parallel state, scalar prior, FutureSeed content map, or Sudoku-specific rule.

## 3. Mechanism Hypothesis

Hard global constraints may require evidence accumulated under one address
direction to remain available under another later in the same scan. Diagonal
decay cannot perform that transport. A learned, content-conditioned orthogonal
rotation of the live KxV state should increase address reuse without changing
state norm or adding a state bank. If stable live transport is the missing
operation, it should improve hardest-range late-loop correction and exact
closure while keeping state magnitude comparable to the parent.

The falsifiable prediction is stronger than activation: the rotation angle and
state residual must vary across boards and heads, the transform must preserve
Frobenius norm at every boundary, the post-boundary state read and full model
output must change, terminal
state magnitude must remain bounded, and one registered exact-quality route
must pass. Active rotations with no exact gain close this mechanism. A norm or
identity failure closes the implementation before science.

## 4. Candidate

The sequence is split only at the pinned official FLA GDN2 native chunk size
`64`. The first 64 tokens run through the unchanged official `chunk_gdn2` from
the inherited state. A normalized mean of the first-chunk per-head outputs is
mapped by one per-layer head-shared `V32 -> (2*K32 + 1)` linear controller:

- the first two K32 vectors are orthonormalized into a state-row plane;
- the final scalar gives `theta = pi * tanh(raw_theta)`;
- the KxV state is rotated exactly in that two-dimensional plane;
- the remaining 17 tokens continue through unchanged official `chunk_gdn2`
  from the rotated state;
- the two official output segments are concatenated normally.

The angle row is zero-initialized. Therefore `theta=0`, `sin(theta)=0`, and
`cos(theta)=1` at migration; the rotation is an exact identity regardless of
the randomly initialized plane. The plane rows become learnable after the angle
opens, which requires an explicit two-stage gradient contract. The transform
is orthogonal for every signed angle and preserves state Frobenius norm by
construction.

There is no extra token, scan, persistent state, reverse traversal, output
residual, second core, search, or Sudoku logic. The existing 81-token official
scan is exposed as its native 64+17 chunks, with one state transform between
them. The controller adds exactly
`12 * 32 * (2*32 + 1) = 24,960` parameters and zero recurrent-state values.
All 12 layers participate because the rotation can affect both model output and
terminal FutureSeed state.

## 5. Frozen Configuration

- D256/L12/H8/K32/V32, channel multiplier4
- position-QK addressing, random traversal
- native terminal FutureSeed, loop5, equal CE at every loop
- 12 pinned official-FLA GDN2/Triton layers
- one head-shared V32-to65 boundary controller per layer
- one orthogonal live-state transform at native token64 boundary
- official full-diversity curriculum and exact step3000 parent
- BF16, effective batch128, optimizer/RNG/data order exact resume
- LR0.0015, weight decay0.001, seed52
- candidate-only step3000-to3100 continuation

The frozen control is not rerun. There is no angle range, controller width,
plane source, boundary count/location, state scale, seed, optimizer, loss,
batch, width/depth, or duration follow-up.

## 6. Launch Gates

Before formal execution, one strict GPU1 CUDA contract must prove:

1. exactly CUDA index0 and the registered UUID;
2. pinned FLA source SHA `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. 24 official `chunk_gdn2` calls and 24
   `ChunkGDN2FunctionBackward` graphs in a full native-FutureSeed pass;
4. exact 24,960-parameter migration and no new recurrent state;
5. bit-exact full output and all 12 terminal states at zero angle, including
   synthetic nonzero incoming states, against the unsplit parent path;
6. finite nonzero first-stage gradients in all 12 angle rows and, after a
   synthetic angle opening, finite nonzero gradients in all plane rows;
7. exactly 12 active boundary paths, finite nonzero angle/state/output change,
   and board/head variation after opening;
8. plane orthogonality error and boundary norm-ratio error each `<=1e-5` in
   FP32, with changed state when chunk summary or incoming state is shuffled;
9. exact-resume optimizer/RNG/data-order migration and complete step3001
   checkpoint plus metrics JSON;
10. no fallback, NaN, OOM, source/data drift, hidden CPU model path, or
    accidental Sudoku/traversal-boundary dependency.

Any contract or production-probe miss closes this implementation. It does not
authorize a tolerance relaxation, different chunk boundary, angle cap, plane
source, extra transform, or custom non-official recurrent fallback.

### Contract Result

Source SHA `1994db5918bc8783210cbc5c146ecf44f53a4cd1` was pushed, read
back, and checked out as clean detached worktree
`/huyang2/double-loop/worktrees/p-gdn3-009-1994db5`. Contract R1
exited before importing Torch because the wrapper selected the repository
metadata venv rather than the established training interpreter. It is archived
as a non-science abort; no model or CUDA path was entered.

R2 used the unchanged source and `/opt/conda/bin/python`. The zero-angle full
output and all terminal-state identities passed, including all 12 synthetic
nonzero incoming-state checks. The registered recurrent graph gate then failed:
each final terminal state exposed one `ChunkGDN2FunctionBackward`, not the
required two, yielding `[1,1,1,1,1,1,1,1,1,1,1,1]`. The externally split
official calls therefore do not expose the required trainable two-chunk state
chain under this composition. The run closed immediately before the angle and
plane gradient stages. No step3001 probe or formal continuation started.

- R1 log SHA256: `c867578f26e9e4a9a1ebf12cae7d4c2fe21936a33392727c16ef9b42fe69018c`
- R1 abort SHA256: `9e83972912a3a93ed6aab73348593bff10c9858771743a3e48af7ca804836126`
- R2 log SHA256: `3152539a09bdf40d512205ea4b86d1e73600422174e7dddc1ff03a1de19cbbad`
- R2 abort SHA256: `3b7721d2cebb29d10af4caba910ae664d4d7144faacd86270281545c26347ebb`

## 7. Science, Stability, And Cost Gates

At step3100, activation and stability require all of:

- exactly 12 boundary paths active;
- mean absolute angle `>=1e-4`, finite board/head variation, and plane
  non-degeneracy;
- rotated-state residual relative RMS and post-boundary state-read relative RMS
  each finite and `>=1e-4`;
- mean/max boundary Frobenius-norm ratio within `[0.9999, 1.0001]`;
- terminal state RMS finite at every loop and no more than `4x` the frozen
  control at the matching loop;
- no fallback.

Quality passes by exactly one route:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with every hard
   range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

The scan performs the same 81 official token transitions but incurs a second
Python/kernel boundary and the rotation controller. Before any result, warmed
elapsed overhead is capped below `40%` and peak allocated memory overhead below
`30%` versus the frozen control. OOM, unstable timing, or either ceiling miss
kills the run regardless of quality.

Any activation, stability, quality, or cost miss discards this mechanism. Do
not rescue angle range, controller width, plane source, boundary count/location,
state scale, seed, LR, loss, batch, model width/depth, or duration.

## 8. Required Readout

Report and archive:

- mixed and official51-55/56-60/61-64 loop1-5 exact/blank/wrong cells;
- train CE and same-board loop3-to5 correction counts;
- angle magnitude/variation, plane error, norm ratio, state/read residual,
  terminal-state RMS, and all-12 path activity;
- independently warmed throughput, peak allocation/reservation, and timing
  stability;
- config, source, parent, checkpoint, metrics, log, and comparison hashes;
- same-board loop1-5 visualization using frozen-control case IDs.

## 9. Decision

Discard P-GDN3-009 at the strict CUDA contract. Forward composition is exactly
parent-compatible at zero angle, but the registered two-chunk recurrent
backward chain is absent. Do not rescue the graph assertion, boundary, custom
backward, angle/controller/plane source, state scale, seed, optimizer, loss,
batch, model size, or duration. A successor must remain inside one
differentiable official recurrent call, or introduce a separately justified
official-compatible transition whose state gradient contract is proven before
any continuation.
