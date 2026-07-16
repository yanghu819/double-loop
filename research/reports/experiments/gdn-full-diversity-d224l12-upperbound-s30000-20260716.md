# Native FutureSeed Data-Compute Upper Bound

## 1. Metainfo

- Plan ID: `P-SCALE-034`
- Status: in progress; first gate targets exact step10000
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

Pending GPU1 launch from exact step8000.

## 7. Artifacts And Visualization

Each leg will archive config, launch environment, logs, score, fixed checkpoint
eval, official eval, source SHA, abort/rollover metadata, case JSON/HTML, and a
scaling-curve dashboard. Checkpoints, models, datasets, and source snapshots
remain outside Git.

## 8. Submission Record

No tag unless the primary score is at least `0.50` and the scaling conclusion
is clean.
