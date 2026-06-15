# h120 Resume Continuation D256 Loop6 Scale Run

Run: `h120resume-cont-d256-l12-loop6-s14300-20260615T0105Z-0e999dc`

Source SHA: `0e999dc770a1cb8a7ecf305f39582357981936cb`

Recorded: `2026-06-15T02:30:04.019370+00:00`

## Hypothesis

The previous resumable run proved that the clean D256/L12/loop6 path can be resumed from a step10000 training checkpoint, but its observed h120 curve only reached the old step9800 point. This continuation tests the higher-value question: does genuinely later hard-stage compute still improve h120, or was the clean path already near a practical ceiling?

This is not a new mechanism ablation. It is a direct compute-scaling readout for fixed FutureSeed plus recurrent loops.

## Configuration

- Board: 12x12, random holes, implicit 3x4 boxes.
- Resume point: `/huyang2/double-loop/.worktrees/h120resumable-d256-l12-loop6-s14300-20260614T1217Z-0e999dc/runs/h120resumable-d256-l12-loop6-s14300-20260614T1217Z-0e999dc/checkpoints/latest.pt`.
- Model: D256/L12, heads8, head_dim32, channel_mult4, loop6, `LOOP_LOSS=all`.
- FutureSeed: fixed update, decay `0.0`, no scratch state, no loop feedback, no loop-time conditioning.
- Noise: `noise_scale=0.0`, `scratch_noise_scale=0.0`, `rollout_noise_scale=0.0`.
- Kernel/runtime: GPU1 only, CUDA visible device 0, bf16, RWKV statepassing.
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:8500`.
- Batch/eval: train batch72, eval512, rollout loop values `1,3,6`, checkpoint eval holes `96,108,120,132`.
- Train checkpoints: saved every 1000 steps and at eval checkpoints; checkpoints remain remote and are not tracked by Git.

## Artifacts

- Local metadata: `config.json`, `metadata.json`, `score.json`, `logs/run.log`.
- Local eval outputs: `output/checkpoint_eval_step*.json`, `output/futureseed_loop_seed52.json`, `output/futureseed_loop_seed52.md`, `output/futureseed_loop_case_seed52.html`.
- Source provenance: `source_HEAD.txt`, empty `source.patch`, `source_snapshot.ls.txt`, `source_snapshot.ls.txt.sha256`.
- Remote final checkpoint: `/huyang2/double-loop/.worktrees/h120resume-cont-d256-l12-loop6-s14300-20260615T0105Z-0e999dc/runs/h120resume-cont-d256-l12-loop6-s14300-20260615T0105Z-0e999dc/checkpoints/latest.pt`.
- Checkpoints are not tracked by Git.

## Results

The run resumed from saved train step10000 and completed through step14300. Final train CE was `0.0975`, train total loss was `0.2439`, and train time for the continuation was `5239.8s`.

| checkpoint | h96 loop6 exact | h108 loop6 exact | h120 loop3 exact | h120 loop6 exact | h120 loop6 blank_acc | h132 loop6 exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10800 | 0.9922 | 0.9180 | 0.0547 | 0.0918 | 0.6886 | 0.0000 |
| 11800 | 1.0000 | 0.9121 | 0.0469 | 0.0977 | 0.7044 | 0.0000 |
| 12800 | 0.9922 | 0.9199 | 0.0449 | 0.1055 | 0.7064 | 0.0000 |
| 13800 | 0.9961 | 0.9121 | 0.0762 | 0.1094 | 0.7042 | 0.0000 |
| 14300 | 0.9941 | 0.9336 | 0.1191 | 0.1621 | 0.7418 | 0.0000 |
| final full eval | 0.9980 | 0.9473 | 0.1172 | 0.1777 | 0.7465 | 0.0000 |

Final h120 loop curve:

| loop | exact | valid_sudoku | blank_acc |
| --- | ---: | ---: | ---: |
| 1 | 0.0000 | 0.0000 | 0.4147 |
| 3 | 0.1172 | 0.1211 | 0.7309 |
| 6 | 0.1777 | 0.1895 | 0.7465 |

The K1 rollout oracle gap is zero at loop6, so this result still does not point toward selector work.

## Insight

Clean compute was still useful after the earlier step9800 point. h120 loop6 exact moved from the prior `0.0586` checkpoint to `0.0918`, then `0.0977`, `0.1055`, `0.1094`, and finally `0.1621` at step14300, with the final full eval reaching `0.1777`.

The gain was not smooth: there was a low-slope stretch from step10800 to step13800, then a late jump at step14300. That matters because a short continuation would have undercalled this direction. The path is slow, but not dead.

Loop is still the solve mechanism. On h120, loop1 exact is still `0.0`, loop3 is `0.1172`, and loop6 reaches `0.1777`. The model is not solving h120 with one stronger forward pass; it is using recurrence to assemble boards.

h132 remains fully closed, with loop6 exact `0.0` and blank accuracy about `0.156`. The current recipe has opened h120 but has not reached a new frontier beyond it.

## Decision

This run upgrades the clean h120 score from `0.0723` to `0.1777`, so the clean FutureSeed+loop scaling path still has real headroom. Starting from scratch is now low ROI; if spending more pure compute, resume from this continuation checkpoint rather than replaying the first 14300 steps.

The next high-ROI choice is either a further genuine continuation to test whether h120 approaches a supported regime, or one simple late-loop FutureSeed state-update change that lets later loops keep correcting boards. Do not spend budget on selector work, Sudoku repair, scratch noise/Gaussian sweeps, deeper loop-count repeats, or same-budget width repeats.

No experiment tag was created because the primary score is below `0.50`.
