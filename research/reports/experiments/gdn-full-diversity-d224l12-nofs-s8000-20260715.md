# Matched Full-Diversity No-FutureSeed Control

## 1. Metainfo

- Plan ID: `P-SCALE-033`
- Status: in progress, step1000 scored and exact step1100 rollover preserved
- Planned: 2026-07-15 CST
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Experiment source SHA: `503f367b4df84509feabe33e8c40edaeb5c2fb63`
- Training mode: from scratch with exact resumable checkpoints

## 2. Mechanism Hypothesis

The strongest D224 result jointly uses FutureSeed, 3.83M independent boards,
five recurrent loops, and CE supervision at every loop. It proves that the full
system scales, but does not isolate which part supplies the opening advantage.

FutureSeed is intended to place a learned summary of future tokens into a
left-to-right recurrent state before the normal scan. If that mechanism is
useful, disabling only this state should make the otherwise matched GDN learn
hard boards later, require more recurrent compute, or fail to open. If the
no-FutureSeed model follows the same curve, FutureSeed cannot be the paper's
core causal claim.

## 3. Intervention

- Match D224/L12/H14/D16, GDN expand-v4, and BF16 Triton recurrent kernels.
- Match loop5 with equal CE supervision at every loop.
- Match effective batch128 and the 3,831,994 independent-board split.
- Match curriculum `46-50:100,51-55:5900,51-64:2000` through step8000.
- Change only `FUTURE_SEED_SCALE=1` to `0`; keep the parameter path present.
- Evaluate holes53/60/64 at steps 1000, 3000, 4500, 6000, 7000, and 8000.
- Report loop1-to-loop5 exact/blank, wall time, and peak GPU memory.
- Add no noise, scratch, repair, search, selector, or Sudoku-specific rule.

## 4. Prediction, Budget, And Kill Criteria

This is one seed and one arm, not an ablation table. The maximum budget is about
five GPU hours across exact-checkpoint leases. The first lease runs only as far
as can be safely checkpointed before platform shutdown; no CPU smoke is used.

The FutureSeed claim is supported if the matched no-FS curve is materially
slower or weaker, with a hard exact gap of at least `0.03`, or if no-FS needs at
least `20%` more compute to reach the same score. If no-FS reaches the same
exact, blank accuracy, and speed, the core claim is rejected.

Do not stop at step1000 because full-diversity learning is delayed. Stop at the
predeclared step4500 gate only if holes53/60/64 loop5 exact are all below `0.02`
and blank accuracies are all below `0.55`. Stop immediately on wrong GPU, dirty
source, malformed checkpoint, NaN, or OOM. Do not rescue a miss with seed, LR,
loss, width, schedule, or noise sweeps.

## 5. Paper Decision

- no-FS stays closed: FutureSeed supplies a necessary cheap future-context
  initialization under this compute budget;
- no-FS opens later but reaches parity: FutureSeed is an optimization and
  compute-efficiency mechanism;
- no-FS matches the curve: FutureSeed is not the main cause, and the paper must
  pivot to data plus recurrent supervision rather than claim future context.

## 6. Results

### GPU Smoke And First Lease

The one-step GPU smoke completed a real BF16 Triton forward/backward pass,
four-way gradient accumulation, optimizer update, checkpoint write, GPU eval,
and visualization. It confirmed `git_dirty=0`, `fs_state_norm=0`, and no CPU
fallback, NaN, or OOM.

The formal run is
`gdn-full-diversity-d224l12-nofs-s8000-20260715T1235Z-503f367`. It used
`43,760 MiB` on GPU1. An initial launcher attempt correctly refused to run
because smoke had refreshed tracked leaderboard/index files; those two
smoke-generated files were restored to source SHA before the successful launch.
No model code or configuration changed.

Early matched training CE is:

| Step | no-FS CE | FS CE | no-FS minus FS | no-FS loop5-loop1 | FS loop5-loop1 |
|---:|---:|---:|---:|---:|---:|
| 100 | 1.6789 | 1.6346 | +0.0443 | +0.0016 | +0.0012 |
| 200 | 1.8222 | 1.6090 | +0.2132 | +0.0003 | +0.0005 |
| 300 | 1.7550 | 1.2472 | +0.5078 | +0.0012 | -0.0083 |
| 400 | 1.7298 | 1.0966 | +0.6332 | -0.0010 | -0.0169 |
| 500 | 1.6806 | 1.0725 | +0.6081 | -0.0010 | -0.0256 |

This is already a meaningful opening diagnostic: without FutureSeed, local CE
learning continues, but it is much slower after the hard stage begins. By
step300-500 the FS model also gets an increasingly useful reduction from later
loops, while the no-FS loop reduction remains nearly zero. This supports the
mechanism prediction that future initialization makes recurrent computation
usable earlier. It is not yet a final accuracy conclusion; no fixed hard-bucket
evaluation is scheduled before step1000, and delayed scaling remains possible.

The first lease ran from `2026-07-15T12:55:56Z` to `13:20:34Z`. The watchdog
waited for a complete post-guard checkpoint, terminated only Python PID `4081`,
and wrote `lease_rollover.json` for exact step500. GPU1 returned to zero usage;
GPU2 remained halted. The next leg resumes optimizer, scheduler, and RNG state
from `train_state_step000500.pt` and targets the predeclared step1000 gate.

### Step1000 Matched Gate

The exact-resume leg
`gdn-full-diversity-d224l12-nofs-resume500-s1000-20260715T1325Z-503f367`
restored model, optimizer, scheduler, and RNG state from step500. The no-FS CE
remained on a high plateau while the matched FS curve had already opened:

| Step | no-FS CE | FS CE | no-FS minus FS |
|---:|---:|---:|---:|
| 600 | 1.6754 | 1.0061 | +0.6693 |
| 700 | 1.6792 | 0.9871 | +0.6921 |
| 800 | 1.6676 | 0.9865 | +0.6811 |
| 900 | 1.6653 | 0.9688 | +0.6965 |
| 1000 | 1.6580 | 0.9971 | +0.6609 |

The fixed step1000 holes53/60/64 results are:

| Holes | no-FS exact | FS exact | no-FS blank | FS blank | FS blank gain |
|---:|---:|---:|---:|---:|---:|
| 53 | 0.0000 | 0.0176 | 0.2657 | 0.5145 | +0.2488 |
| 60 | 0.0000 | 0.0215 | 0.2677 | 0.5254 | +0.2577 |
| 64 | 0.0000 | 0.0215 | 0.2671 | 0.5250 | +0.2579 |

This is a large causal opening difference from one mechanism switch. The no-FS
model is not merely missing full-board exact: its per-blank accuracy is about
chance-like and its five loops are neutral. Mixed blank accuracy is
`0.2653/0.2663/0.2650/0.2644/0.2647` across loops1-5, with exact zero at every
loop. Official range loop5 exact is also zero for `46-50`, `51-55`, and
`56-64`, with blank accuracy `0.3896`, `0.2699`, and `0.2542` respectively.

The visual case bank confirms the same failure shape rather than hiding a metric
artifact. In hard `56-64` cases, wrong cells change only `34->33->33`,
`37->34->34`, `34->35->35`, and `39->37->36` from loops1->3->5; duplicate
conflicts remain essentially fixed. There are no solved or almost-solved cases.
No search, repair, selector, noise, or Sudoku rule is active.

The training runner interprets the sum of `HOLE_STAGES` as the global endpoint;
`FULL_STEPS` does not truncate a staged resume. The first resumed process thus
continued through a complete step1100 checkpoint before it was stopped by exact
PID and recorded in `abort.json`. A zero-train evaluation was then run from the
exact step1000 checkpoint by setting stage counts to sum to 1000. Future resumes
must make stage counts sum to the intended global target. No samples or optimizer
updates were lost.

Per the predeclared rule, step1000 is not a final verdict because full-diversity
training previously showed delayed acceleration. The next scored gate remains
step3000. The current result already supports the narrower claim that FutureSeed
materially improves optimization and sample efficiency under matched compute;
whether no-FS eventually catches up remains open.

## 7. Artifacts And Visualization

The step1000 archive includes config, score, logs, source SHA, fixed evaluation,
loops1/3/5 case-bank HTML, the matched FS-vs-no-FS comparison page, and the
resume `abort.json`. Metadata-light archive SHA256 values are:

- first step0-500 leg: `87b61692abefaddf42d1bc84c13d5385361d06dc227c0fd4850dfea907d1cde7`
- step500-1100 resume: `53bbdc2547d4b8beed0f47b329117ecfce29a822e8ade74d8f84a78288cf93d4`
- exact step1000 eval and visualization: `4f74ad628dd1d77cda63c48029c3989d4cf369c5704b4897198837eba4049e2c`

Later scored checkpoints will use the same artifact contract. Intermediate
lease rollovers archive an exact checkpoint and `lease_rollover.json` or
`abort.json`; checkpoints themselves remain outside Git.

## 8. Submission Record

No tag unless the mechanism conclusion is strong and the primary score is at
least `0.50`.
