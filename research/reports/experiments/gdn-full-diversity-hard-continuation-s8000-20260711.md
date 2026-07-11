# Full-Diversity Hard Continuation To Step8000

## 1. Metainfo

- Plan ID: `P-SCALE-030`
- Status: in-progress
- Planned: 2026-07-12 05:46 CST / 2026-07-11T21:46:00Z
- Launched: 2026-07-12 05:56 CST / 2026-07-11T21:56:15Z
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Experiment source SHA: `eeb38f5b6f9b6c1b1df0bbd3ca149daec06a171d`
- Resume checkpoint: P-SCALE-029 exact step6000
- Run: `gdn-full-diversity-hard-resume6000-s8000-20260711T2150Z-eeb38f5`
- Launch PID / active Python PID: `1398` / `46147`
- Launch script SHA256: `8759592f0f71a9fa08498e3354e2ae46d908660f4ca8dcdcf0347152e1a50274`
- Runtime check: CUDA-only resume confirmed at global step6000; A800 allocation
  `45116 MiB`, Triton recurrent mode, BF16.
- Provenance note: the detached worktree is exact source SHA. Its only tracked
  difference is the generated `runs/visualization_index.html`; the launcher
  fails if any other tracked path differs. A new worktree/checkout was attempted
  first but blocked in remote filesystem I/O, so no source-code dirtiness was
  accepted to recover time on the active GPU lease.

## 2. Mechanism Hypothesis

P-SCALE-029 established a delayed but strong independent-data scaling crossover:
full-diversity D224/L12 reaches overall loop5 exact `0.2500`, official 51-55
`0.3926`, and official 56-64 `0.1270`. Its holes64 checkpoint still has a large
loop1-to-loop5 gain (`0.0254 -> 0.2500`), so recurrent computation remains useful.

The decisive next question is whether the hardest range is still data/compute
limited or whether the current generic recurrent state has saturated. If useful
scaling slope remains, continuing the exact model/optimizer state while widening
hard exposure from 51-55 to 51-64 should improve 56-64 exact without erasing the
51-55 bridge. If hard exact is flat or regresses despite more full-diversity
tokens, stop same-state compute and change generic state efficiency instead.

This is bitter-lesson scaling: more independent data tokens and compute, with no
new task rule, loss, selector, repair, search, noise, or hand-designed solver.

## 3. Intervention

- Resume exact P-SCALE-029 `train_state_step006000.pt`, including model,
  optimizer, scaler, and RNG state.
- Keep native FutureSeed GDN D224/L12/H14/D16, expand-v4, no short conv.
- Keep loop5, every-loop CE, fixed state update, effective batch128, BF16, and
  Triton recurrent kernel.
- Extend global training from step6000 to step8000.
- Add one broad hard-data stage only: `51-64:2000`.
- Checkpoint/eval at step7000 and step8000 on holes53/60/64.
- Final official ranges 46-50, 51-55, 56-64 and case-bank loops 1/3/5.

## 4. Prediction, Budget, And Kill Criteria

Prediction:

- official 56-64 exact should rise from `0.1270` toward `>=0.16`;
- official 51-55 should remain `>=0.38`;
- holes60/64 should preserve large loop1-to-loop5 gains and ideally exceed
  `0.30` exact.

Budget: 2,000 resumed steps, approximately 1.7-1.9 GPU hours plus final eval,
on the current GPU1 lease.

Success:

- primary: official 56-64 `>=0.16` while 51-55 `>=0.38`;
- strong: official 56-64 `>=0.20` or final mixed exact `>=0.32`;
- mechanism: loop5 continues to correct full boards rather than merely raising
  blank accuracy.

Kill:

- stop on wrong GPU, NaN, OOM, malformed checkpoint, or source ambiguity;
- at step7000, stop only if holes60 and holes64 both fall below `0.20` exact and
  blank accuracy shows no gain over the step6000 regime;
- do not respond to a miss by sweeping LR, schedule, seed, loss, width, or loop.

## 5. Paper Decision

Success supports the claim that independent relational data and recurrent
compute continue to scale the FutureSeed state beyond its previous blank-count
cliff. Failure is also decisive: it marks the compute ceiling of the current
D224 state and redirects the next investment to a more efficient generic state
representation, not Sudoku-specific correction.

## 6. Results

The step7000 gate is a clean continuation signal:

| Bucket | step6000 loop5 exact / blank | step7000 loop1 exact | step7000 loop3 exact | step7000 loop5 exact / blank | loop1-to-loop5 gain |
|---|---:|---:|---:|---:|---:|
| holes53 | `0.2109 / 0.6344` | `0.0176` | `0.1777` | `0.2480 / 0.6727` | `+0.2305` |
| holes60 | `0.2402 / 0.6643` | `0.0215` | `0.1895` | `0.2832 / 0.7003` | `+0.2617` |
| holes64 | `0.2500 / 0.6410` | `0.0254` | `0.2012` | `0.2617 / 0.6698` | `+0.2363` |

All three hard buckets improve in blank accuracy and full-board exact. The
largest exact gains are holes53 `+0.0371` and holes60 `+0.0430`; holes64 still
gains `+0.0117`. This passes the continuation rule decisively: holes60 and
holes64 are both above `0.20`, and blank accuracy rises materially. Loop compute
also remains the dominant source of board-level correction rather than merely
copying loop1.

The first leg was proactively stopped after the complete step7000 checkpoint
and evaluation because the remaining GPU1 lease could not safely fit another
1000 steps plus final evaluation. Exact PIDs `1455/1454/1453/1398` were stopped,
GPU1 returned to `0 MiB`, and `lease_rollover_step7000.json` records the
transition at `2026-07-11T22:59:50Z`. The next leg will resume exact model,
optimizer, and RNG state from step7000; no experiment setting changes.
