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

The run is still in progress. Fixed holes53/60/64 quality decisions begin at
step1000; the predeclared mechanism decision remains at step4500 or later.

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

The second-lease watchdog PID is `585`. It begins rollover protection at
`2026-07-13T08:40:07Z`, after the planned step6000 evaluation window, and has a
hard deadline of `2026-07-13T09:00:07Z`. It writes
`lease_rollover_2.json` only after a complete checkpoint and stops the exact
resume and monitor PIDs.

Every scored checkpoint will archive fixed-bucket metrics; the final decision
checkpoint must include loops1/3/5 hard-case visualizations.

## 9. Submission Record

No tag unless score reaches `>=0.50` and the scaling claim is clean.
