# P-GDN3-008: Terminal-State Consolidation Sweep

## 1. Metainfo

- Status: discarded; activation and cost gates passed, both registered quality
  routes failed
- Date: 2026-08-07
- Branch: `codex/gdn3-terminal-consolidation-20260807`
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

P-FS3-001/002/003 and P-GDN3-005/006/007 all activate their registered
mechanisms without improving hard exact. They close analytic FutureSeed
residuals, learned producer compression/routing, scalar erase/write coupling,
an independent recurrent expert, and a controller that reads the incoming
layer state before the official scan.

The source audit exposes one untested boundary. P007 computes
`Q_unit @ S_in` once from the state received from the previous layer, then
precomputes all K/V/erase/write residuals before the token scan. It does not
allow the current layer to revise its own terminal state after that state has
seen the complete sequence. P008 tests that missing transition operation. It
is not another transfer-content map, output readout, parallel state, scalar
prior, or Sudoku-specific bias.

## 3. Mechanism Hypothesis

A single causal GDN2 pass must simultaneously accumulate evidence and commit
the memory that native FutureSeed transfers to the next layer. Early writes
cannot use evidence that arrives later in that same pass. If hard closure is
limited by this commit order, a second same-order state transition initialized
from the first terminal state should consolidate globally informed memory and
improve late-loop board correction.

This predicts a nonzero, board- and token-varying correction address; a
nonzero change in terminal state after the correction sweep; and stronger
loop3-to5 hard-board correction. If the transition is active but exact and
same-board correction do not improve, post-scan consolidation is not the local
bottleneck and the mechanism is closed without changing its source, scale,
scan count, or duration.

## 4. Candidate

The parent position-QK official-GDN2 pass is unchanged and still supplies the
model output. On producer layers0-10, its token outputs before output
normalization feed one per-layer, head-shared, zero-initialized V32-to-K32
linear map. This produces `K_corr` for one second pinned-official
`chunk_gdn2` transition that:

- starts from the first pass terminal state;
- reuses the first pass V payload and erase/write gates;
- uses zero correction decay;
- runs in the same traversal order;
- returns only the consolidated terminal state for native FutureSeed.

At migration `K_corr=0`, so every correction update is exactly zero and the
second terminal state equals the first terminal state. The model output is
never replaced or augmented by correction output. After learning, each token
edits a state that already contains the complete first-pass sequence, and the
correction state evolves token by token within the official recurrence.

Layer11 has no downstream receiver and therefore has no refiner or unused
parameter. The mechanism adds exactly `11 * 32 * 32 = 11,264` parameters, no
new persistent state, no reverse scan, no second independent core, no search,
and no Sudoku logic. It adds one official correction scan in each of 11
producer paths. This fixed operation is the candidate itself, not a scan-count
or model-depth sweep.

## 5. Frozen Configuration

- D256/L12/H8/K32/V32, channel multiplier4
- position-QK addressing, random traversal
- native terminal FutureSeed, loop5, equal CE at every loop
- 12 pinned official-FLA GDN2/Triton parent layers
- 11 head-shared V32-to-K32 correction projections
- one same-order correction transition on producer layers0-10
- official full-diversity curriculum and exact step3000 parent
- BF16, effective batch128, optimizer/RNG/data order exact resume
- LR0.0015, weight decay0.001, seed52
- candidate-only step3000-to3100 continuation

The frozen control is not rerun. There is no correction source, output scale,
scan count, layer subset, seed, optimizer, loss, batch, width, depth, or
duration follow-up.

## 6. Launch Gates

Before formal execution, one strict GPU1 CUDA contract must prove:

1. exactly CUDA index0 and the registered UUID;
2. pinned FLA source SHA `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. 12 parent and 11 correction official `chunk_gdn2` paths, with 23
   `ChunkGDN2FunctionBackward` graphs in a full native-FutureSeed pass;
4. exact 11,264-parameter migration and no unexpected parameter/state;
5. bit-exact full output and all 12 terminal states at zero correction K;
6. per-layer exact output/state identity under synthetic nonzero incoming
   state, including correction paths0-10;
7. finite nonzero gradients in all 11 correction projections from downstream
   native FutureSeed;
8. finite nonzero correction address, board/token variation, terminal-state
   residual, and within-sweep state progression after a synthetic opening;
9. head-permutation equivariance and changed terminal state when first-pass
   token outputs or terminal state are shuffled across boards;
10. exact-resume optimizer/RNG/data-order migration and complete step3001
    checkpoint plus metrics JSON;
11. no fallback, NaN, OOM, source/data drift, or hidden CPU model path.

Any contract or production-probe miss closes this implementation. It does not
authorize an output residual, projection-source change, fewer/more sweeps, or
layer subset.

Contract R1 on pushed SHA
`6ea0a12c86edf07854416cadda30c6990a336c61` exited before exercising the
candidate transition. The diagnostics dictionary referenced
`terminal_consolidation_diag` in the position-QK path, while the call producing
that value had been inserted by an ambiguous patch anchor into the mutually
exclusive fast-slow path. The resulting `NameError` occurred on the first
control forward; no identity, official-kernel, gradient, activation, science,
or cost assertion ran or failed. PGID28464 exited naturally, allocations
cleared, and the log plus non-science abort are retained under
`/huyang2/double-loop/artifacts/launch/p-gdn3-008/`. R2 moves only that existing
call to its registered position immediately after the first position-QK
official chunk. The mechanism, parameters, parent, configuration, prediction,
and every gate remain unchanged.

Contract R2 passed on exact pushed SHA
`605ae881ed87fb1987a0d859f247cb84b93ec4c2`. The clean detached worktree
reported the registered GPU UUID and pinned FLA source, exact parameter delta
`11,264`, 12 parent official GDN2 modules, and 23
`ChunkGDN2FunctionBackward` paths. Full model output and all 12 terminal states
were exact at zero initialization, including synthetic nonzero incoming state.
All 11 projections had finite nonzero gradient. The opened synthetic path had
correction-K relative RMS `0.001966`, terminal-state residual relative RMS
`0.004743`, finite board/token variation, exact head-permutation equivariance,
and explicit dependence on shuffled first-pass outputs and terminal state. The
contract log SHA256 is
`3421b399696e8ada61eddfa909bec76ac17587d3d3b93e193a17a8b410ace3ee`.

The exact-resume production probe
`p-gdn3-008-terminal-consolidation-probe-s3001-20260807T045208Z-605ae88`
completed with status0 from the registered step3000 parent. Optimizer, RNG,
data-order and source migration were accepted, and the complete step3001
checkpoint and metrics JSON were written. At loop5, all 11 producer refiners
were enabled; correction-K relative RMS/batch std/token std was
`0.019985/0.000842/0.006540`, while terminal residual relative RMS/batch std
was `0.129669/0.008143`. Metrics/checkpoint SHA256 are
`2c30576f17804afe3ee7ecf160597dd3c60ab4865b8f1fb0d7172903f573db43` and
`3714a2648229543c1ae098d0ebb70923644559adbddfdfe050ed1171805603d5`.
This probe establishes activation and production fit only; its eight-board
score is not a science readout.

The sole formal candidate
`p-gdn3-008-terminal-consolidation-s3100-20260807T045713Z-605ae88` completed
with status0 from the same parent and exact source after GPU clearance. Its
wrapper PGID was `30169`, Python child `30250`, and launch log is
`/huyang2/double-loop/artifacts/launch/p-gdn3-008/formal-20260807T045713Z-605ae88.log`.
No frozen-control rerun or concurrent evaluation was started. The process
released the registered GPU naturally after endpoint artifact completion.

## 7. Science And Cost Gates

At step3100, activation requires all of:

- exactly 11 producer correction paths active;
- correction K relative RMS and board/token variation each finite and
  `>=1e-4`;
- terminal residual relative RMS and board variation finite and `>=1e-4`;
- finite nonzero within-sweep state progression and correction-projection RMS;
- no fallback.

Quality passes by exactly one route:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with every hard
   range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

The second official state transition is the intended compute cost. Before any
result, warmed elapsed overhead is capped below `120%` and peak allocated
memory overhead below `80%` versus the frozen control. OOM, timing instability,
or either ceiling miss kills the run regardless of quality.

Any activation, quality, or cost miss discards this mechanism. Do not rescue
projection source/scale, correction decay/gates, scan count, layer subset,
seed, LR, loss, batch, model width/depth, or duration.

## 8. Required Readout

Report and archive:

- mixed and official51-55/56-60/61-64 loop1-5 exact/blank/wrong cells;
- train CE and same-board loop3-to5 correction counts;
- correction address, board/token variation, terminal residual, state
  progression, and projection-weight activation;
- independently warmed throughput, peak allocation/reservation, and timing
  stability;
- config, source, parent, checkpoint, metrics, log, and comparison hashes;
- same-board loop1-5 visualization using frozen-control case IDs.

## 9. Decision

Discard P-GDN3-008. The mechanism and both cost gates pass, but neither
registered quality route passes. Do not rescue projection source/scale,
correction decay/gates, scan count, layer subset, seed, LR, loss, batch,
width/depth, or duration.

### 9.1 Endpoint score

Train CE improves from `0.858618` to `0.855832`, but exact closure does not
follow. Official loop5 control/candidate exact and blank accuracy are:

| range | control exact / blank | candidate exact / blank | blank delta |
|---|---:|---:|---:|
| 51-55 | `0.001953 / 0.573766` | `0.003906 / 0.575090` | `+0.001325` |
| 56-60 | `0 / 0.503861` | `0 / 0.505714` | `+0.001853` |
| 61-64 | `0 / 0.591923` | `0 / 0.596532` | `+0.004609` |

The hard51-64 macro loop5 exact score changes only
`0.000651 -> 0.001302` (`+0.000651`), far below the registered `+0.02`
primary gate. Mixed loop5 exact is unchanged at `0.025391`, so the `+0.03`
alternate gate also fails. Mixed loop1-to5 exact is
`0.015625/0.023438/0.025391/0.025391/0.025391`; blank accuracy is
`0.511905/0.537324/0.542918/0.544212/0.543373`, versus control
`0.515087/0.536904/0.544841/0.545540/0.545610`.

Official candidate loop1-to5 exact/blank trajectories are:

| range | exact loops1-5 | blank loops1-5 |
|---|---|---|
| 51-55 | `0/0/0/0.003906/0.003906` | `0.531092/0.564601/0.573837/0.574625/0.575090` |
| 56-60 | `0/0/0/0/0` | `0.474280/0.498988/0.504169/0.505233/0.505714` |
| 61-64 | `0/0/0/0/0` | `0.501603/0.578951/0.594091/0.595769/0.596532` |

On the same 256 boards per hard range, candidate mean wrong cells across
loops1-to5 are `25.719/24.254/23.953/23.934/23.863` (51-55),
`29.645/28.449/28.109/28.027/27.977` (56-60), and
`31.941/26.922/26.137/26.016/26.039` (61-64). Candidate loop3-to5
correction is stronger than control on 51-55 (`0.0898` versus `-0.0156`) and
56-60 (`0.1328` versus `0.0664`), but weaker on 61-64 (`0.0977` versus
`0.1836`). The alternate same-board requirement therefore also fails.

### 9.2 Activation and stability

All 11 producer paths are active. At loop5, correction-K relative RMS,
board std, and token std are `0.120164/0.004845/0.036855`, and projection
weight RMS is `0.013016`. The live correction transition is therefore not a
silent identity.

However, the unconstrained terminal transition is unstable across loops.
Terminal residual relative RMS grows from `1.1981e4` at loop1 to
`4.7109e8` at loop5; loop5 state board std and correction-output RMS reach
`1.0649e10` and `3.7959e9`. Values remain finite and downstream FutureSeed
unit normalization prevents a runtime failure, but the state magnitude is not
a reusable scalable memory representation. This instability is a mechanism
failure, not grounds for a post-result scale rescue.

### 9.3 Cost and provenance

Fresh-process continuation throughput changes from `15.4969` to `12.1511`
effective boards/s. Elapsed time is `825.970 -> 1053.403` seconds
(`+27.54%`); peak allocated memory is `13186.42 -> 16213.28` MiB
(`+22.95%`), and reserved memory is `14288 -> 17426` MiB (`+21.96%`). These
are inside the preregistered `+120%` time and `+80%` allocation ceilings.

Endpoint artifact SHA256 values are:

- metrics: `04d680e17bfc96ed26df9a2a8198cd344fd4e480959314aa05b588699e68a21e`;
- checkpoint: `08894f95282fa96e75a65df6353e158dcfe6313d7f90957811eeae5da882d081`;
- config: `462895db3a736228b4fbfdfa709c71cd9c0a661b55f8601a81d148c34c477f97`;
- run log: `db2a361ec0c764ba6e197f00f58d7327161d9af6084826ed6476b82979a4a126`;
- formal log: `f432ef085523676f573dab7ba91658756ee6f88ac467412d9f90b1d9ea0e8505`;
- source snapshot: `4ddc57967b02788378c9aedb52f7569a212ca26f39ba4afab6ce95257857f1f0`;
- comparison JSON/HTML:
  `22ad476dd23171fd7b6c107ce402a4ee02cacceacdd61621360ca7dd263d7b40` /
  `05bbf381ab68dfafc717cf49461d57e408fb99171cd05f61ccb63b8ec8afbf27`.

The remote comparison is
`/huyang2/double-loop/runs/p-gdn3-008-comparison-20260807T051700Z-605ae88`.
The archived same-board visualization is
`research/reports/visualizations/gdn3-terminal-consolidation-20260807/comparison.html`.

### 9.4 Mechanism conclusion

P008 provides the first positive evidence that a post-scan transition can
move all three hard-range blank accuracies in the desired direction. It also
shows that a fixed second pass with unnormalized repeated writes does not
convert that movement into exact closure and produces an unusably large state.
The next candidate must change the live recurrent transition with an explicitly
stable, scalable state update; another fixed sweep, pre-scan residual, parallel
expert, scalar prior, or transfer-content router is not justified.
