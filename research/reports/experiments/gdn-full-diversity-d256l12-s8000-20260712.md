# Full-Diversity D256 Joint Scaling Test

## 1. Metainfo

- Plan ID: `P-SCALE-032`
- Status: in progress
- Planned: 2026-07-12 CST
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Experiment source SHA: `79fcd7d09b038f64ae86045ae180ac7c1999f91c`
- Training mode: from scratch, resumable exact checkpoints
- Fit run: `gdn-full-diversity-d256l12-fit-step1-20260712T124933Z-79fcd7d`
- Formal run: `gdn-full-diversity-d256l12-s8000-20260712T125118Z-79fcd7d`

## 2. Mechanism Hypothesis

P-SCALE-028 showed that D256/L12 does not beat D224/L12 when both are trained on
only 1,000 source boards repeated through augmentations. P-SCALE-029 then showed
that 3.83M independent source boards sharply improve D224 after a delayed
crossover around step4500.

The unresolved scaling question is therefore joint rather than one-dimensional:
does the larger D256 backbone need independent relational data to use its extra
capacity? If yes, D256 should start no better than D224 but eventually overtake
the D224 full-diversity curve. If it does not, ordinary width is not an effective
capacity-per-compute axis for the current FutureSeed-GDN state formulation.

## 3. Intervention

- Train native FutureSeed GDN D256/L12/H16/D16 from scratch.
- Keep `GDN_EXPAND_V=4.0`, no short convolution, and fixed FutureSeed/update.
- Keep loop5 with equal CE supervision on every loop.
- Use BF16 Triton recurrent forward/backward and effective batch128.
- Use all 3,831,994 independent training boards.
- Match the successful D224 curriculum exactly:
  `46-50:100,51-55:5900,51-64:2000`.
- Evaluate holes53/60/64 at steps 1000, 3000, 4500, 6000, 7000, and 8000.
- Add no noise, scratch state, gate, repair, search, selector, or Sudoku rule.

## 4. Prediction, Budget, And Kill Criteria

Budget: approximately 6-7 GPU hours across resumable GPU1 leases. A single
GPU-only optimizer-step fit gate precedes the formal run; no CPU smoke is used.

Prediction and gates:

- do not quality-stop at step1000;
- at step4500, stop only if all hard-bucket exact values are below `0.07`, all
  blank accuracies are below `0.60`, and the curve is not accelerating;
- primary step6000: holes53 exact `>=0.21` and holes60 or holes64 `>=0.25`;
- continue to step8000 only if D256 reaches or clearly approaches the D224
  full-diversity step6000 curve;
- strong step8000: beat D224 in at least two fixed buckets and improve official
  51-55 `0.4473` or 56-64 `0.1621`.

Stop immediately on wrong GPU, source ambiguity, NaN, OOM, or malformed
checkpoint. Do not rescue a miss with width, seed, LR, loss, or schedule sweeps.

## 5. Paper Decision

Success supports joint data/model scaling: FutureSeed-GDN can use more generic
capacity when trained on enough independent relations. Failure establishes D224
as the more compute-efficient frontier and redirects work toward a more
efficient generic recurrent state, not more parameters or task-specific tricks.

## 6. Results

### GPU Fit And Launch

- The GPU-only fit completed one real BF16 Triton forward/backward, four-way
  gradient accumulation, optimizer update, checkpoint write, and GPU eval.
- Fit step1 CE was `2.3988`; the expected random-initialization holes53 exact
  was `0.0000`. The fit wrote `train_state_step000001.pt` without NaN or OOM.
- The formal detached worktree was separately created at the exact source SHA
  and verified with zero tracked changes before launch.
- Formal training PID: `3544`; CUDA worker PID at launch: `3597`.
- GPU monitor PID: `3543`; proactive lease watchdog PID: `4257`.
- Peak observed allocation through step100: `52,006 MiB` on GPU1. GPU2
  remained halted.
- Step100 CE was `1.8322` (`loop1=1.8338`, `loop5=1.8322`). The matched D224
  reference had CE `1.6346` at step100, so D256 begins slower. This is a risk
  diagnostic, not an early-stop condition; the hypothesis predicts a delayed
  crossover.
- `train_state_step000100.pt` was written successfully. Measured startup-to-
  checkpoint throughput was approximately `3.7 s/step`.

The run remains in progress through the approved step8000 tail. Fixed
holes53/60/64 decisions have now been completed through step6000.

### First Lease And Exact Resume

- The first lease completed through step2600. The checkpoint-safe watchdog
  stopped CUDA worker PID `3597` immediately after
  `train_state_step002600.pt` was fully written, then archived
  `lease_rollover_1.json` at `2026-07-12T15:12:20Z`.
- Step1000 had CE `1.1277`, holes53 exact `0.0000`, and blank accuracy
  `0.4599`. The matched D224 reference was CE `0.9971`, exact `0.0176`, and
  blank accuracy `0.5145`, so D256 remained behind at the first fixed gate.
- This did not trigger a stop: the experiment explicitly forbids a quality
  decision at step1000 because the tested hypothesis is a delayed crossover.
- Training CE reached `0.8812` at step2600; loop1 CE was `1.0089` while loop5
  CE was `0.8812`, so recurrent depth was already reducing supervised loss.
- GPU1 was restarted on `2026-07-13`; it remained queued for approximately 25
  minutes. Kimi WebBridge and the AIStation API agreed on Pending/Running state,
  while GPU2 remained halted.
- The exact step2600 checkpoint, optimizer state, scheduler state, and RNG were
  restored in run
  `gdn-full-diversity-d256l12-resume2600-s8000-20260713T053631Z-79fcd7d`.
  The resumed training PID is `263`; its GPU monitor PID is `262`.
- A missing container-level `jq` binary caused the first resume preflight to
  exit before training. The launcher now parses rollover JSON with Python's
  standard library; no dependency was installed and training semantics did not
  change.

### Step3000 Fixed Evaluation

The three fixed buckets at step3000 were:

| Holes | Loop1 exact | Loop5 exact | Loop gain | Loop5 blank acc | D224 loop5 exact |
|---:|---:|---:|---:|---:|---:|
| 53 | 0.0137 | 0.0176 | +0.0039 | 0.5512 | 0.0273 |
| 60 | 0.0117 | 0.0293 | +0.0176 | 0.5667 | 0.0332 |
| 64 | 0.0137 | 0.0371 | +0.0234 | 0.5534 | 0.0430 |

D256 remains below D224 in all three exact buckets, so this is not a positive
scaling result yet. However, it closed most of the large step1000 deficit by
step3000, and harder buckets show larger loop1-to-loop5 gains. That is the
predicted shape of a delayed optimization crossover and justifies continuing
to the predeclared step4500 gate without changing the configuration.

### Step4500 Gate And Second Lease Rollover

| Holes | Loop1 exact | Loop5 exact | Loop gain | Loop5 blank acc | D224 loop5 exact |
|---:|---:|---:|---:|---:|---:|
| 53 | 0.0156 | 0.0488 | +0.0332 | 0.5940 | 0.1055 |
| 60 | 0.0137 | 0.0586 | +0.0449 | 0.6157 | 0.1172 |
| 64 | 0.0234 | 0.0801 | +0.0566 | 0.5968 | 0.1230 |

Step4500 did not trigger the predeclared kill rule: holes64 exact exceeded
`0.07`, holes60 blank accuracy exceeded `0.60`, and loop gain continued to
grow with difficulty. D256 was still clearly behind D224, so this was only a
continue signal, not positive evidence for width scaling.

The second lease watchdog stopped only the exact run after the complete
`train_state_step005800.pt` write and recorded `lease_rollover_2.json` at
`2026-07-13T08:41:03Z`. GPU1 was restarted on 2026-07-15 after remaining in
the AIStation queue; GPU2 stayed halted. The exact optimizer, scheduler, and
RNG state were then restored from step5800 for the predeclared step6000 gate.

### Step6000 Delayed-Crossover Gate

| Holes | Loop1 exact | Loop5 exact | Loop gain | Loop5 blank acc | D224 loop5 exact | Delta vs D224 |
|---:|---:|---:|---:|---:|---:|---:|
| 53 | 0.0176 | 0.2031 | +0.1855 | 0.6332 | 0.2109 | -0.0078 |
| 60 | 0.0215 | 0.2305 | +0.2090 | 0.6583 | 0.2402 | -0.0097 |
| 64 | 0.0215 | 0.2109 | +0.1895 | 0.6303 | 0.2500 | -0.0391 |

Step6000 CE is `0.5635`. The strict primary numbers were narrowly missed:
holes53 did not reach `0.21`, and holes60/64 did not reach `0.25`. However,
the separately declared continuation rule was to fund step8000 if D256 reached
or clearly approached the D224 step6000 curve. D256 is now within one exact
percentage point on holes53 and holes60 after trailing badly at step4500.

More importantly, D256's step4500-to-step6000 exact gains are
`+0.1543/+0.1719/+0.1308`, versus D224 gains of
`+0.1054/+0.1230/+0.1270`. This is the first clean evidence of the predicted
late acceleration. It justifies the original 6000-to-8000 hard tail without
changing data, optimizer, loss, noise, or model semantics. It does not yet
establish that D256 is better per compute; the final test is whether step8000
beats D224 in at least two fixed buckets or improves the official ranges.

## 7. Conclusions

Pending.

## 8. Artifacts And Visualization

Remote launch artifacts are under
`/huyang2/double-loop/artifacts/launch/pscale032/`; checkpoints are under
`/huyang2/double-loop/models/gdn-full-diversity-d256l12-s8000-20260712T125118Z-79fcd7d/checkpoints/`.

The watchdog begins rollover protection at `2026-07-12T15:11:27Z`, waits for
the next fully written 100-step checkpoint, terminates only this run's exact
Python PID, writes `lease_rollover_1.json`, and stops the exact GPU monitor PID.
Its hard deadline is `2026-07-12T15:26:27Z`, before the current lease expires.

The second-lease watchdog PID was `585`. It entered rollover protection at
`2026-07-13T08:40:07Z`, preserved a complete step5800 checkpoint, wrote
`lease_rollover_2.json`, and stopped only the exact resume and monitor PIDs.
The step5800-to-step6000 resume is
`gdn-full-diversity-d256l12-resume5800-s6000-20260715T093650Z-79fcd7d`.

Every scored checkpoint will archive fixed-bucket metrics; the final decision
checkpoint must include loops1/3/5 hard-case visualizations.

## 9. Submission Record

No tag unless score reaches `>=0.50` and the scaling claim is clean.
