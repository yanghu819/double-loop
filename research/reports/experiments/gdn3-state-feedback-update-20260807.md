# P-GDN3-007: Closed-Loop State-Feedback Update

## 1. Metainfo

- Status: static implementation and preregistration; no GPU run authorized yet
- Date: 2026-08-07
- Branch: `codex/gdn3-state-feedback-update-20260807`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent mechanism: D256/L12 position-QK GDN3 plus native terminal FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen matched control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

No GPU model process may run from a local-only commit. The implementation,
checker, launcher, configuration, and this preregistration must be pushed
first. GPU execution must use a clean detached worktree at that exact SHA.

## 2. Evidence Boundary

Five active matched mechanisms now fail to improve hard exact:

1. orthogonal FutureSeed innovation residual;
2. shared producer-update compression;
3. address-local producer-update routing;
4. coherent erase/write scalar coupling;
5. a complete independent recurrent residual expert.

The first three alter transfer content, the fourth alters one aggregate update
degree of freedom, and the fifth adds a private address/update/state stack.
P-GDN3-006 proves that a second state can become active without changing the
main solver's board decisions; its parallel official-GDN2 expert more than
doubles elapsed time and still weakens the two hardest late-loop corrections.
This closes another transfer residual, scalar prior, or expert-width/count
experiment.

## 3. Mechanism Hypothesis

The remaining structural gap is closed-loop state interaction. In the current
position-QK path, official GDN2 reads its incoming KxV state, but current write
address, payload, erase, and write gates are projected from position/current
content before the kernel. The carried state does not directly condition the
decision that edits that state.

Hard iterative correction may require each token to compare its proposed edit
with the memory already stored at its current query. If this is the missing
mechanism, a state-derived controller should become board- and token-dependent,
alter all four write components, and improve late-loop full-board closure. If
it activates but exact and same-board correction remain unchanged, then
open-loop update control is not the local bottleneck and this direction is
closed without hidden-size or scale rescue.

## 4. Candidate

Before the unchanged pinned official `chunk_gdn2`, each receiving layer uses
its normalized position query to read the incoming state:

`R[b,t,h,v] = Q_unit[b,t,h,k] @ S_in[b,h,k,v]`.

One head-shared controller per layer maps V32 to hidden16 with SiLU, then to
four zero-initialized residuals: K32 write-address, V32 payload, K32 erase-logit,
and V32 write-logit. The effective official-kernel inputs are:

- `K' = K + dK`;
- `V' = V + dV`;
- `b' = sigmoid(b_raw + db)`;
- `w' = sigmoid(w_raw + dw)`.

The final projection is exactly zero, so parent output and recurrent states are
bit-exact at migration, including nonzero incoming states. The controller is
shared across heads and therefore head-permutation equivariant. It adds exactly
2,560 parameters per layer and 30,720 total, no recurrent state, no extra scan,
no second FLA layer, and no Sudoku-specific logic. Layer0 has no incoming
FutureSeed and remains exactly inactive; layers1-11 are the registered receiving
paths.

## 5. Frozen Configuration

- D256/L12/H8/K32/V32, channel multiplier4
- position-QK addressing, random traversal
- native terminal FutureSeed, loop5, equal CE at every loop
- 12 pinned official-FLA GDN2/Triton layers
- head-shared V32->16->128 state-feedback controller per layer
- official full-diversity curriculum and exact step3000 parent
- BF16, effective batch128, optimizer/RNG/data order exact resume
- LR0.0015, weight decay0.001, seed52
- candidate-only step3000-to3100 continuation

The frozen terminal continuation is not rerun. There is no controller hidden
size, output scale, layer subset, feedback target, seed, optimizer, loss,
duration, width, or depth follow-up.

## 6. Launch Gates

The strict GPU1 CUDA contract must prove:

1. exactly CUDA index0 and the registered UUID;
2. pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. 12 exact official `GatedDeltaNet2` layers and 12
   `ChunkGDN2FunctionBackward` terminal graphs;
4. exact 30,720-parameter migration and no unexpected parameter;
5. bit-exact full-model output and all 12 terminal states at zero output
   projection;
6. per-layer exact output/state identity under synthetic nonzero incoming
   state;
7. finite nonzero gradient in all 12 zero output projections, then finite
   nonzero gradient in all 12 input and output projections after one synthetic
   opening step;
8. exactly 11 active receiving layers in a full native-FutureSeed pass;
9. finite nonzero state-read RMS, between-board variation, token variation,
   and K/V/b/w relative changes after opening;
10. head-permutation equivariance and a changed controller output when incoming
    states are shuffled across boards;
11. exact-resume optimizer/RNG/data-order migration and a complete step3001
    checkpoint plus metrics JSON;
12. no fallback, NaN, OOM, source drift, data drift, or hidden CPU model path.

A failed contract or production probe closes this implementation. It does not
authorize a smaller controller or target subset.

## 7. Science And Cost Gates

At step3100, activation requires all of:

- mean enabled receiving-layer fraction exactly `11/12`;
- state-read RMS and between-board std finite and nonzero;
- controller residual relative RMS and token std `>=1e-4` and finite;
- K, V, erase, and write relative changes each finite and nonzero;
- all 11 receiving layers active, no fallback.

Quality passes by exactly one of:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with every hard
   range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

Because this adds no second recurrent kernel or recurrent state, independent
warmed elapsed time and peak allocated memory overhead must each remain below
`25%` versus the frozen control. Timing instability, OOM, or fallback kills
the run regardless of quality.

Any activation, quality, or cost miss discards this exact state-feedback
controller. Do not rescue controller hidden size, output scale, target subset,
layer sharing, seed, LR, loss, batch, model width/depth, or duration.

## 8. Required Readout

Report and archive:

- mixed and official51-55/56-60/61-64 loop1-5 exact/blank/wrong cells;
- train CE and same-board loop3-to5 correction counts;
- state-read, board/token variation, K/V/b/w change, and controller-weight
  activation metrics;
- independently warmed throughput, peak allocation/reservation, and timing
  stability;
- config, source, parent, checkpoint, metrics, log, and comparison hashes;
- same-board loop1-5 visualization using the frozen control case IDs.

## 9. Decision

Pending pushed-source static checks, strict CUDA contract, and exact-resume
step3001 probe. No formal run is authorized before every launch gate passes.
