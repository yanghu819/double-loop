# Native FutureSeed Data-Compute Upper Bound

## 1. Metainfo

- Plan ID: `P-SCALE-034`
- Status: in progress; step24000 gate passed, final conditional gate is step30000
- Planned: 2026-07-16 11:05 CST / 2026-07-16T03:05:36Z
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Parent checkpoint: P-SCALE-030 exact step8000 FutureSeed-GDN train state
- Maximum endpoint: step30000, conditional on checkpoint slope

## 2. Question And Hypothesis

The project is deliberately keeping native FutureSeed plus recurrent loop as
the mainline and is not spending this budget on a bidirectional baseline. The
question is now the upper bound of the current scalable formulation.

P-SCALE-029 and P-SCALE-030 established that independent relational data and
hard-token compute are the only tested axes with consistent positive return.
The step8000 model has consumed about `8000 * 128 = 1,024,000` nominal board
draws, while the full-diversity train split contains `3,831,994` rows. Because
training samples with replacement, this is not a count of unique examples, but
it shows that the previous endpoint used substantially less than one
dataset-equivalent draw budget.

Hypothesis: the current D224 native FutureSeed state is still data/compute
limited. Extending broad 51-64 training without any architecture or objective
change should improve full-board exact, primarily through loop4/5. If two
successive gates fail to exceed the step8000 best, pure scaling of this state
has reached its useful ceiling and the next investment must change the generic
state update rather than add Sudoku logic.

## 3. Intervention

- Resume exact `train_state_step008000.pt`, including model, optimizer,
  scheduler, and RNG state.
- Keep D224/L12/H14/D16 GDN expand-v4 and native FutureSeed scale1.
- Keep loop5, equal CE at every loop, effective batch128, BF16 Triton recurrent,
  fixed loop update, and the same 3,831,994-row train split.
- Continue only the existing broad hard range 51-64.
- First leg: step8000 to10000; second decision gate: step12000.
- Continue to16000/20000/24000/30000 only while the measured slope remains
  useful. Step30000 is approximately `3.84M` nominal board draws.
- Every gate evaluates fixed holes53/60/64. Every completed leg exports mixed
  loop1-5, official 46-50/51-55/56-64, and hardest-case HTML.
- Add no noise, gate, loss, width, depth, loop count, seed, solver, repair,
  search, selector, or task rule.

## 4. Prediction, Budget, And Kill Criteria

Baseline at step8000:

| Readout | Value |
|---|---:|
| mixed loop5 exact | `0.2715` |
| fixed holes53/60/64 loop5 exact | `0.2559 / 0.3066 / 0.2852` |
| official 51-55 exact | `0.4473` |
| official 56-64 exact | `0.1621` |

The first step10000 result is diagnostic and is not killed for one noisy point.
At step12000, stop if both step10000 and step12000 fail to improve any of
holes60/64, mixed exact, or official56-64 by at least `+0.02` over step8000 and
blank accuracy/CE also show no positive slope. After a successful step12000,
stop only after two successive later gates fail to set a `+0.01` new best on
mixed or official hard exact.

Primary upper-bound success is mixed exact `>=0.35`, or official56-64
`>=0.22` while official51-55 remains `>=0.45`. Strong success is mixed exact
`>=0.40` or official56-64 `>=0.30`. Stop immediately on wrong GPU, dirty
source, malformed resume, NaN, OOM, or an unverified checkpoint.

The first leg costs about two GPU hours plus evaluation. The complete ladder
can cost roughly twenty GPU hours across exact-checkpoint leases, but later
legs are conditional rather than prepaid.

## 5. Paper Decision

Positive scaling supports the claim that FutureSeed plus recurrent compute has
not saturated at the previous quarter-dataset-equivalent budget. A flat ladder
sets an empirical upper bound for the current state formulation. Either result
is useful; no miss will be rescued with a low-information sweep.

## 6. Results

The step10000 and step12000 gates both completed from exact train-state
checkpoints on GPU1. The model, optimizer, scheduler, RNG, data split, loop
count, loss, and FutureSeed formulation remained unchanged.

| Readout | step8000 | step10000 | step12000 | step12000 vs 10000 |
|---|---:|---:|---:|---:|
| mixed loop5 exact | `0.2715` | `0.3066` | `0.3730` | `+0.0664` |
| holes53 loop5 exact | `0.2559` | `0.3008` | `0.3340` | `+0.0332` |
| holes60 loop5 exact | `0.3066` | `0.3320` | `0.3789` | `+0.0469` |
| holes64 loop5 exact | `0.2852` | `0.3027` | `0.3574` | `+0.0547` |
| official 51-55 exact | `0.4473` | `0.4824` | `0.5352` | `+0.0527` |
| official 56-64 exact | `0.1621` | `0.1855` | `0.2285` | `+0.0430` |

The primary upper-bound gate passes in both independent ways: mixed exact is
above `0.35`, and official 56-64 is above `0.22` while 51-55 remains above
`0.45`. The strong `mixed >=0.40` or `56-64 >=0.30` gate is not yet reached.

The mechanism signal is stronger than the aggregate score. At step12000,
mixed exact across loops1-5 is
`0.0234 / 0.0664 / 0.2617 / 0.3496 / 0.3730`. Loop1 remains at the old
operating point while loop5 improves by `+0.1016` over step8000, so added hard
compute is primarily teaching later recurrent iterations to close the board.
On official 56-64, exact moves from `0.0` at loop1 to `0.2285` at loop5.

Selected 56-64 cases show concrete correction rather than mask expansion:

- 57 blanks: `31 -> 4 -> 0` wrong cells across loops1/3/5.
- 58 blanks: `30 -> 9 -> 0`.
- Almost solved: `15 -> 7 -> 1`.
- Hard failure: `29 -> 12 -> 5`.

Decision: continue the same clean scaling ladder to the conditional step16000
gate. Every tracked hard metric set a new best by more than `+0.01`, so the
later-gate stop rule has not fired. Do not add noise, a new loss, more loops,
width, repair, search, or selector.

The step16000 gate completed on 2026-07-19 from the exact step15425 train-state
checkpoint after an AIStation lease rollover. This leg changed no model,
optimizer, data, objective, or loop setting.

| Readout | step12000 | step16000 | Delta |
|---|---:|---:|---:|
| mixed loop5 exact | `0.3730` | `0.3945` | `+0.0215` |
| holes53 loop5 exact | `0.3340` | `0.3809` | `+0.0469` |
| holes60 loop5 exact | `0.3789` | `0.4180` | `+0.0391` |
| holes64 loop5 exact | `0.3574` | `0.3828` | `+0.0254` |
| official 51-55 exact | `0.5352` | `0.5605` | `+0.0254` |
| official 56-64 exact | `0.2285` | `0.2598` | `+0.0312` |

Mixed exact across loops1-5 is
`0.0234 / 0.0586 / 0.2812 / 0.3828 / 0.3945`. The extra training still acts
through loops3-5 rather than moving loop1, so this remains direct evidence for
learned recurrent refinement. The official 56-64 aggregate is `0.2598`; the
nearby `0.2656` number from the visualization case bank is not used as the
primary aggregate. The step15425-to16000 segment took `1953.4s` and peaked at
`44527.1MB` allocated memory.

Decision: the predeclared `+0.01` later-gate rule passes again. Pause this
ladder while P-LA-001 performs the higher-information GDN2/KDA state-edit
comparison, then resume unchanged to step20000 if neither alternative produces
a stronger return per unit compute.

The step20000 gate completed on 2026-07-21 from the exact step16000 train-state
checkpoint. The full model, optimizer, scheduler, RNG state, data split,
objective, and five-loop computation were preserved.

| Readout | step16000 | step20000 | Delta |
|---|---:|---:|---:|
| mixed loop5 exact | `0.3945` | `0.4375` | `+0.0430` |
| holes53 loop5 exact | `0.3809` | `0.4238` | `+0.0430` |
| holes60 loop5 exact | `0.4180` | `0.4395` | `+0.0215` |
| holes64 loop5 exact | `0.3828` | `0.3945` | `+0.0117` |
| official 51-55 exact | `0.5605` | `0.5840` | `+0.0234` |
| official 56-64 exact | `0.2598` | `0.3125` | `+0.0527` |

Mixed exact across loops1-5 is
`0.0234 / 0.0840 / 0.3242 / 0.4160 / 0.4375`. Loop1 again stays fixed while
loops3-5 improve, so the additional training is teaching recurrent closure
rather than a stronger first pass. On selected official 56-64 boards, wrong
cells change `33 -> 5 -> 0`, `30 -> 12 -> 0`, `24 -> 11 -> 1`, and
`27 -> 8 -> 5` across loops1/3/5. This is direct cell-level correction, not
an aggregate-only effect.

The formal official loop5 exact/blank results are `1.0000/1.0000` for 46-50,
`0.5840/0.8351` for 51-55, and `0.3125/0.6942` for 56-64. Nearby case-bank
estimates are visualization samples and are not used as primary metrics. The
step16000-to20000 segment took `13759.8s` and peaked at `44527.1MB` allocated
and `44600MB` reserved GPU memory.

Decision: every tracked hard metric again sets a new best and the strong
upper-bound gate is now reached in both forms (`mixed >=0.40` and official
56-64 `>=0.30`). Continue the unchanged clean ladder to step24000. Do not add
width, loops, noise, a new loss, task rules, repair, search, or selector.

The step24000 gate completed on 2026-07-22 before the GPU1 lease expired. It
resumed the exact step20000 train state and completed training, fixed-checkpoint
evaluation, official evaluation, case banks, visualization, and run recording.

| Readout | step20000 | step24000 | Delta |
|---|---:|---:|---:|
| mixed loop5 exact | `0.4375` | `0.4648` | `+0.0273` |
| holes53 loop5 exact | `0.4238` | `0.4258` | `+0.0020` |
| holes60 loop5 exact | `0.4395` | `0.4629` | `+0.0234` |
| holes64 loop5 exact | `0.3945` | `0.4414` | `+0.0469` |
| official 51-55 exact | `0.5840` | `0.6387` | `+0.0547` |
| official 56-64 exact | `0.3125` | `0.2949` | `-0.0176` |

Mixed exact across loops1-5 is
`0.0234 / 0.1055 / 0.3477 / 0.4414 / 0.4648`. Loop1 remains unchanged, while
the loop5 gain over loop1 increases to `+0.4414`. Selected official 56-64
boards change `32 -> 7 -> 0`, `32 -> 11 -> 0`, `31 -> 3 -> 0`, and
`29 -> 7 -> 5` wrong cells across loops1/3/5. This remains real recurrent
correction even though the hardest aggregate fluctuates downward.

The formal official loop5 exact/blank results are `1.0000/1.0000` for 46-50,
`0.6387/0.8513` for 51-55, and `0.2949/0.6826` for 56-64. The 56-64 case-bank
estimate is `0.3008`; it is not the primary aggregate. Training took
`13791.6s` and peaked at `44527.1MB` allocated and `44600MB` reserved.

Decision: this gate passes the predeclared rule through new mixed, holes60,
holes64, and official51-55 bests, but it also supplies the first warning that
the hardest official distribution may be near a noisy plateau. Run only the
predeclared final step30000 endpoint with the identical formulation. Continue
pure scaling only if step30000 sets a new mixed best and recovers or improves
official56-64; otherwise stop this state formulation without a rescue sweep.

## 7. Artifacts And Visualization

Each leg archives config, launch environment, logs, score, fixed checkpoint
eval, official eval, source SHA, abort/rollover metadata, case JSON/HTML, and a
scaling-curve dashboard. Checkpoints, models, and datasets remain outside Git.
The step20000 run additionally commits a compact relevant-source snapshot for
the exact detached SHA; it contains no checkpoint, model, dataset, or cache.

Completed step12000 run:

- Remote run:
  `/huyang2/double-loop/.worktrees/pscale034-upperbound-1c99589-20260716/runs/gdn-full-diversity-d224l12-upperbound-s12000-lease-20260716T131753Z-1c99589`
- Exact checkpoint:
  `/huyang2/double-loop/models/gdn-full-diversity-d224l12-s6000-20260711T1510Z-eeb38f5/checkpoints/train_state_step012000.pt`
- Remote metadata archive:
  `/huyang2/double-loop/artifacts/pscale034-step12000-metadata-light.tgz`
- Local interactive dashboard:
  `runs/gdn-full-diversity-d224l12-upperbound-s12000-lease-20260716T131753Z-1c99589/index.html`

Completed step16000 run:

- Remote run:
  `/huyang2/double-loop/.worktrees/pscale034-upperbound-f8e009b-20260717/runs/gdn-full-diversity-d224l12-upperbound-s16000-lease-20260719T025124Z-f8e009b`
- Exact checkpoint:
  `/huyang2/double-loop/models/gdn-full-diversity-d224l12-s6000-20260711T1510Z-eeb38f5/checkpoints/train_state_step016000.pt`
- Remote metadata archive:
  `/huyang2/double-loop/artifacts/gdn-full-diversity-d224l12-upperbound-s16000-lease-20260719T025124Z-f8e009b-metadata-light.tgz`
- Local interactive dashboard:
  `runs/gdn-full-diversity-d224l12-upperbound-s16000-lease-20260719T025124Z-f8e009b/visualizations/index.html`

Completed step20000 run:

- Remote run:
  `/huyang2/double-loop/.worktrees/pscale034-step20000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s20000-lease-20260721T110943Z-f8e009b`
- Exact checkpoint:
  `/huyang2/double-loop/models/gdn-full-diversity-d224l12-s6000-20260711T1510Z-eeb38f5/checkpoints/train_state_step020000.pt`
- Exact relevant-source archive:
  `runs/gdn-full-diversity-d224l12-upperbound-s20000-lease-20260721T110943Z-f8e009b/source_snapshot.tar.gz`
  (SHA256 `0c1914100670e2b9b18ffc0a28a4384ab208a031f107755bd6bbd1e6c293f332`)
- Local interactive dashboard:
  `runs/gdn-full-diversity-d224l12-upperbound-s20000-lease-20260721T110943Z-f8e009b/visualizations/index.html`

Completed step24000 run:

- Remote run:
  `/huyang2/double-loop/.worktrees/pscale034-step24000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s24000-lease-20260721T152314Z-f8e009b`
- Exact checkpoint:
  `/huyang2/double-loop/models/gdn-full-diversity-d224l12-s6000-20260711T1510Z-eeb38f5/checkpoints/train_state_step024000.pt`
- Exact relevant-source archive:
  `runs/gdn-full-diversity-d224l12-upperbound-s24000-lease-20260721T152314Z-f8e009b/source_snapshot.tar.gz`
  (SHA256 `0c1914100670e2b9b18ffc0a28a4384ab208a031f107755bd6bbd1e6c293f332`)
- Local interactive dashboard:
  `runs/gdn-full-diversity-d224l12-upperbound-s24000-lease-20260721T152314Z-f8e009b/visualizations/index.html`

## 8. Submission Record

No tag: the scaling conclusion is clean, but the primary score is `0.4648`,
below the `0.50` tag threshold.
