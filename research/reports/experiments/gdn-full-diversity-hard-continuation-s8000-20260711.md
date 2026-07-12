# Full-Diversity Hard Continuation To Step8000

## 1. Metainfo

- Plan ID: `P-SCALE-030`
- Status: done
- Planned: 2026-07-12 05:46 CST / 2026-07-11T21:46:00Z
- Launched: 2026-07-12 05:56 CST / 2026-07-11T21:56:15Z
- Completed: 2026-07-12 08:05 CST / 2026-07-12T00:05:00Z
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Experiment source SHA: `eeb38f5b6f9b6c1b1df0bbd3ca149daec06a171d`
- Resume checkpoint: P-SCALE-029 exact step6000
- First leg: `gdn-full-diversity-hard-resume6000-s8000-20260711T2150Z-eeb38f5`
- Final leg: `gdn-full-diversity-hard-resume7000-s8000-20260711T2308Z-eeb38f5`
- Launch PID / active Python PID: `1398` / `46147`
- Launch script SHA256: `8759592f0f71a9fa08498e3354e2ae46d908660f4ca8dcdcf0347152e1a50274`
- Runtime check: CUDA-only resume confirmed at global step6000; A800 allocation
  `45116 MiB`, Triton recurrent mode, BF16.
- Second lease: restored through Kimi WebBridge without changing the A800 80GB,
  CPU16, image, memory64GB, or shm20GB configuration. Container
  `6hmai8uvrfo6j-0` resumed the exact step7000 checkpoint at
  `2026-07-11T23:09:08Z`; launch script SHA256
  `a014732180b258111bd97bc85b7011d9d0d1f12990c200c71837d12f35887988`.
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

Actual decision: the primary gate passes, but the strong gate does not. This is
positive evidence for continued data/compute scaling, with diminishing but
nonzero marginal returns by step8000. It is not yet evidence that FutureSeed
beats a matched EqR/no-FS baseline.

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

The exact step7000 checkpoint resumed on a fresh, probed GPU1 A800 lease and
completed naturally at step8000. The second checkpoint gate is:

| Bucket | step7000 loop5 exact / blank | step8000 loop1 | step8000 loop3 | step8000 loop4 | step8000 loop5 exact / blank | exact delta |
|---|---:|---:|---:|---:|---:|---:|
| holes53 | `0.2480 / 0.6727` | `0.0176` | `0.1699` | `0.2461` | `0.2559 / 0.6930` | `+0.0078` |
| holes60 | `0.2832 / 0.7003` | `0.0215` | `0.1875` | `0.2852` | `0.3066 / 0.7147` | `+0.0234` |
| holes64 | `0.2617 / 0.6698` | `0.0254` | `0.1895` | `0.2734` | `0.2852 / 0.6886` | `+0.0234` |

The slope remains positive, especially on holes60/64. The mechanism detail is
more informative than the final number alone: loop1 is unchanged from step7000,
and loop3 slightly falls, while loop4/5 improve. Extra hard training therefore
acts mainly through later recurrent correction rather than making the first
prediction stronger.

Final 512-board mixed evaluation:

| Loop | exact | blank accuracy |
|---:|---:|---:|
| 1 | `0.0234` | `0.5605` |
| 2 | `0.0508` | `0.6534` |
| 3 | `0.1875` | `0.6938` |
| 4 | `0.2598` | `0.7049` |
| 5 | `0.2715` | `0.7055` |

Loop1-to-loop5 exact gain is `+0.2481`. Blank accuracy is nearly saturated by
loop4, but loop5 still converts another `1.17%` of boards to fully correct
solutions. This is real recurrent refinement, not repeated output.

Official test ranges meet the primary decision rule:

| Range | step6000 exact | step8000 exact | delta | step8000 blank accuracy |
|---|---:|---:|---:|---:|
| 46-50 | `1.0000` | `1.0000` | `0.0000` | `1.0000` |
| 51-55 | `0.3926` | `0.4473` | `+0.0547` | `0.7900` |
| 56-64 | `0.1270` | `0.1621` | `+0.0351` | `0.6322` |

The primary gate (`56-64 >=0.16` while `51-55 >=0.38`) passes. The strong gate
does not: `56-64` remains below `0.20` and mixed exact remains below `0.32`.
The correct conclusion is useful but diminishing same-state scaling, not an
unbounded compute curve.

Hard-case evidence shows both success and the remaining limit:

- a 64-blank solved case goes `26 wrong -> 4 -> 0` at loops `1/3/5`;
- a 56-blank almost-solved case goes `26 -> 3 -> 1`;
- a 56-blank hard failure still improves `19 -> 9 -> 5`.

The two resumed 1000-step legs record `3312.1 s` and `3324.4 s`, approximately
`1.84 h` total after step6000. Combined with P-SCALE-029, the from-scratch
step8000 trajectory used about `6.35 GPU hours` across leases. Final peak CUDA
memory is `44,527 MB` allocated and `44,600 MB` reserved. The process completed
naturally and GPU1 returned to `0 MiB`.

## 7. Conclusions

P-SCALE-030 is a clean primary success. More full-diversity hard tokens improve
all fixed hard buckets and both official transition ranges without changing the
model or adding solver-specific behavior. The strongest scientific signal is
that the additional step7000-to-8000 gain appears at loop4/5 while loop1 stays
fixed: later recurrent computation is learning more reliable global closure.

The marginal slope is nevertheless smaller after step7000, and the strong gate
is missed. Do not continue the identical D224 state to 10000/12000 merely to
chase a threshold. The next model-scaling test, if funded, should combine the
proven 3.83M-board diversity with a larger generic backbone and enough tokens to
permit a delayed crossover. It must be one planned scale point, not a width or
seed table. A separate matched no-FS/EqR comparison remains mandatory before a
paper claim about FutureSeed itself.

## 8. Artifacts And Visualization

- Metadata-light bundle SHA256:
  `c50416898f83a061b861bdf820ea8bcf7a488aad04afaf507946a29ede60b963`.
- The bundle contains both run legs, configs, launch environments/scripts,
  logs, checkpoint-eval JSON, final score, source SHA/patch, case JSON/HTML, and
  visualizations.
- Checkpoints, datasets, and the `63 MB` source snapshot remain under
  `/huyang2/double-loop` and are not committed to GitHub.
- In-place dashboard:
  `runs/gdn-full-diversity-hard-resume7000-s8000-20260711T2308Z-eeb38f5/index.html`.
- Dashboard QA passed at desktop and `390x844`: no horizontal overflow, all
  case tabs update the embedded visualization, and no new page errors appeared.

## 9. Submission Record

- Final score: `0.271484375` at
  `metrics.eval_clean.loop5.label_exact`.
- Primary scaling gate: passed.
- Strong scaling gate: not passed.
- No experiment tag: project policy requires score `>=0.50`.
