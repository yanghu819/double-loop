# Matched Full-Diversity No-FutureSeed Control

## 1. Metainfo

- Plan ID: `P-SCALE-033`
- Status: complete; stopped at the predeclared step4500 low-ROI gate
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
model is not merely missing full-board exact: its per-blank accuracy remains
low and its five loops are neutral. Mixed blank accuracy is
`0.2653/0.2663/0.2650/0.2644/0.2647` across loops1-5, with exact zero at every
loop. Official range loop5 exact is also zero for `46-50`, `51-55`, and
`56-64`, with blank accuracy `0.3896`, `0.2699`, and `0.2542` respectively.

The sequence-position diagnostic is more directly tied to the mechanism. At
loop5, averaged over holes53/60/64, no-FS blank accuracy rises from `0.2032` in
the first sequence third to `0.2633` in the middle and `0.3340` in the last
third. The same FutureSeed model is nearly position-flat at
`0.5199/0.5223/0.5228`. A left-to-right model without future initialization is
therefore much weaker on early cells, while terminal-state seeding nearly
removes the directional bias. This supports the intended cheap-future-context
mechanism more specifically than aggregate accuracy alone.

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

### Step3000 Matched Gate

The exact step1100 checkpoint was resumed without changing data, optimizer,
model, loop loss, or curriculum. Training reached an exact step3000 checkpoint
and completed all fixed-bucket, official-range, and case-bank evaluations. The
high no-FS CE plateau did not break:

| Step | no-FS CE | FS CE | no-FS minus FS | no-FS loop1 | no-FS loop5 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 1.6580 | 0.9971 | +0.6609 | 1.6593 | 1.6580 |
| 2000 | 1.6455 | 0.8526 | +0.7929 | 1.6469 | 1.6455 |
| 3000 | 1.6379 | 0.7939 | +0.8440 | 1.6391 | 1.6379 |

The matched fixed hard buckets at step3000 are:

| Holes | no-FS exact | FS exact | no-FS blank | FS blank | FS blank gain |
|---:|---:|---:|---:|---:|---:|
| 53 | 0.0000 | 0.0273 | 0.2737 | 0.5788 | +0.3051 |
| 60 | 0.0000 | 0.0332 | 0.2737 | 0.5885 | +0.3148 |
| 64 | 0.0000 | 0.0430 | 0.2755 | 0.5844 | +0.3089 |

No-FS has therefore consumed three times the optimizer steps and still has not
reached the matched FutureSeed step1000 regime. Its final mixed blank accuracy
is `0.2716/0.2723/0.2719/0.2714/0.2713` across loops1-5 and exact remains zero
at every loop. In contrast, matched FutureSeed holes53 improves from loop1 to
loop5 exact `0.0176 -> 0.0273` and blank `0.5174 -> 0.5788`; holes64 improves
exact `0.0254 -> 0.0430` and blank `0.5211 -> 0.5844`.

The sequence-position diagnostic becomes even stronger. Averaged over
holes53/60/64 at loop5, no-FS is `0.2057/0.2685/0.3488` for early/middle/late
sequence thirds. FutureSeed is `0.5840/0.5815/0.5863`. More training raises the
no-FS mean only slightly and does not remove its left-to-right information
imbalance; FutureSeed remains nearly position-flat.

A zero-training GPU1 evaluation of the exact FutureSeed step3000 checkpoint
also put both arms on the same official test procedure. FutureSeed versus no-FS
loop5 exact/blank is `1.0000/1.0000 vs 0.0000/0.4125` on blanks46-50,
`0.0117/0.6560 vs 0.0000/0.2812` on blanks51-55, and
`0.0078/0.5270 vs 0.0000/0.2623` on blanks56-64. This evaluation did not train
or select a checkpoint; it only exported matched statistics and visual cases.

The visual failure shape agrees with the aggregate result. FutureSeed cases
include blanks51-55 `14 wrong -> 0 -> 0` and `16 -> 3 -> 1`, and blanks56-64
`19 -> 4 -> 3` across loops1/3/5. No-FS blanks56-64 cases instead change
`35 -> 32 -> 32`, `35 -> 35 -> 35`, `36 -> 36 -> 35`, and
`35 -> 35 -> 35`; duplicate conflicts are almost fixed. No-FS has no solved or
almost-solved cases in any official blank range.

This is strong mechanism evidence, but the experiment still honors the
predeclared delayed-scaling rule. Continue the same clean no-FS arm to step4500.
If holes53/60/64 exact all remain below `0.02` and blank all remain below `0.55`,
stop rather than rescue the control with a new loss, schedule, seed, or model.

### Step4500 Final Gate

The matched no-FS continuation completed through an exact step4500 checkpoint.
GPU1's development lease expired after the exact step4000 checkpoint, so the
first leg recorded `abort.json` and the second leg restored model, optimizer,
scheduler, and RNG state from that checkpoint. This was an infrastructure
rollover, not an experimental reset. GPU2 remained halted throughout.

The no-FS training curve remained flat while the FutureSeed model continued to
improve:

| Step | no-FS CE | FS CE | no-FS minus FS |
|---:|---:|---:|---:|
| 1000 | 1.6580 | 0.9971 | +0.6609 |
| 3000 | 1.6379 | 0.7939 | +0.8440 |
| 4500 | 1.6372 | 0.6415 | +0.9957 |

The complete step4500 fixed-bucket comparison is:

| Holes | no-FS exact | FS exact | no-FS blank | FS blank | FS blank gain |
|---:|---:|---:|---:|---:|---:|
| 53 | 0.0000 | 0.1055 | 0.2755 | 0.6166 | +0.3412 |
| 60 | 0.0000 | 0.1172 | 0.2777 | 0.6332 | +0.3555 |
| 64 | 0.0000 | 0.1230 | 0.2766 | 0.6133 | +0.3367 |

Averaged over holes53/60/64, FutureSeed loop5 exact/blank is
`0.1152/0.6211`, versus `0.0000/0.2766` without FutureSeed. The no-FS mixed
evaluation remains exact zero for loops1-5, and blank accuracy only changes
`0.2745 -> 0.2760`. FutureSeed changes mixed exact
`0.0234 -> 0.0234 -> 0.0684 -> 0.1055 -> 0.1113` and blank
`0.5396 -> 0.5929 -> 0.6198 -> 0.6270 -> 0.6283`. Later loops become useful
only after FutureSeed has opened the underlying representation.

The positional diagnostic still isolates the intended information mechanism.
At loop5, fixed-bucket early/middle/late blank accuracy is
`0.2085/0.2696/0.3518` without FutureSeed and
`0.6216/0.6193/0.6223` with FutureSeed. More no-FS training did not remove the
large left-to-right bias; terminal-state seeding nearly removes it.

The same checkpoint pair was evaluated with the same official procedure.
FutureSeed versus no-FS loop5 exact/blank is `1.0000/1.0000 vs
0.0000/0.4187` for blanks46-50, `0.1875/0.7250 vs 0.0000/0.2862` for
blanks51-55, and `0.0605/0.5447 vs 0.0000/0.2641` for blanks56-64. Visual
cases show FutureSeed changing `23 wrong -> 0 -> 0` and
`24 wrong -> 0 -> 0` across loops1/3/5. Matched no-FS cases only change
`31 -> 29 -> 29` and `35 -> 33 -> 33`, with duplicate conflicts fixed or
worse. The effect is therefore not a broad local-accuracy artifact.

The two provenance SHAs differ because later commits added experiment records
and visualization support. A direct diff verified that
`study_rwkv_futureseed_loop.py`, `run.sh`, and `scripts/rwkv_maze_probe.py` are
byte-identical between `eeb38f5` and `503f367`; excluding documentation and run
records, the only changed source file is the visualization builder. The core
model and training semantics are therefore matched.

All three no-FS fixed buckets satisfy the predeclared stop condition: exact is
below `0.02` and blank is below `0.55`. P-SCALE-033 stops at step4500. Do not
run no-FS to step6000/8000 or rescue it with a seed, LR, loss, schedule, noise,
or width sweep. The supported claim is that FutureSeed supplies a strong cheap
future-context initialization and makes subsequent recurrent computation
useful under this budget. The remaining paper limitation is equally important:
this does not yet prove a quality-efficiency advantage over an explicit
bidirectional or noncausal baseline. That matched quality/throughput/VRAM test
is the next high-information experiment.

## 7. Artifacts And Visualization

The step1000 archive includes config, score, logs, source SHA, fixed evaluation,
loops1/3/5 case-bank HTML, the matched FS-vs-no-FS comparison page, and the
resume `abort.json`. The comparison page includes CE, loop behavior, fixed hard
buckets, and early/middle/late sequence-position bias. Metadata-light archive
SHA256 values are:

- first step0-500 leg: `87b61692abefaddf42d1bc84c13d5385361d06dc227c0fd4850dfea907d1cde7`
- step500-1100 resume: `53bbdc2547d4b8beed0f47b329117ecfce29a822e8ade74d8f84a78288cf93d4`
- exact step1000 eval and visualization: `4f74ad628dd1d77cda63c48029c3989d4cf369c5704b4897198837eba4049e2c`
- step1100-to3000 no-FS run: `7be7c9088a11d5d6a52b1d24e744b94abf645f1e8fa050a0a3788748e5acdbc5`
- zero-training FutureSeed step3000 visualization eval: `35a3360820ef3585834e814298bf75a394dc1f5e06c44ed3842ad74a4e76cc31`
- step3000-to4500 no-FS continuation, including the step4000 lease rollover:
  `b7c1f2081e0a3ad7ef06fd58ca51d8c3abcd9a945a690a435d87defd3444db6d`
- zero-training FutureSeed step4500 visualization eval:
  `639e69d1a570121399e5e9a6d107ca1422f578e63dbfddb0982b7abfe086e1e2`

The step3000 matched comparison page includes the complete 100-3000 CE curves,
loop-level blank accuracy, sequence-position bias, fixed and official bucket
tables, and direct links to FutureSeed solved/almost-solved cases and no-FS
hard failures.

The final step4500 comparison page includes the full decision, training CE,
loop-level exact, fixed and official buckets, positional bias, and direct links
to solved, almost-solved, and hard-failure cases. Intermediate lease rollovers
archive an exact checkpoint and `lease_rollover.json` or `abort.json`;
checkpoints themselves remain outside Git.

## 8. Submission Record

No tag unless the mechanism conclusion is strong and the primary score is at
least `0.50`.
