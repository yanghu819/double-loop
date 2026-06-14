# h120 Resumable D256 Loop6 Scale Run

Run: `h120resumable-d256-l12-loop6-s14300-20260614T1217Z-0e999dc`

Source SHA: `0e999dc770a1cb8a7ecf305f39582357981936cb`

Started: `2026-06-14T12:17Z`

Stopped: `2026-06-14T15:28:54Z`

## Hypothesis

Previous clean D256/L12/loop6 h120 runs showed a reproducible but shallow positive hard-stage slope. This run tests whether the same clean FutureSeed+loop path keeps improving when training is resumable across AIStation leases.

This is not an ablation. It answers one decision question: is more clean compute still a useful scaling lever for the current paradigm, or has h120 become a late-loop/global-consistency ceiling that needs a simple FutureSeed state-update change?

## Configuration

- Board: 12x12, random holes, implicit 3x4 boxes.
- Model: D256/L12, heads8, head_dim32, channel_mult4, loop6, `LOOP_LOSS=all`.
- FutureSeed: fixed context, no mutable scratch, no loop-residual update, no feature noise.
- Kernel/runtime: GPU1 only, CUDA visible device 0, bf16, RWKV statepassing.
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:8500`.
- Batch/eval: train batch72, eval512, rollout loops `1,3,6`, checkpoint eval holes `96,108,120,132`.
- Eval checkpoints: steps `6800,7800,8800,9800,10800,11800,12800,13800,14300`.
- Resumability: train checkpoints saved every 1000 steps and at eval checkpoints under the remote run-local `checkpoints/` directory.

## Artifacts

- Local small artifacts: `config.json`, `abort.json`, `logs/run.log`, `output/checkpoint_eval_step*.json`, `source_HEAD.txt`.
- Remote checkpoint resume point: `/huyang2/double-loop/.worktrees/h120resumable-d256-l12-loop6-s14300-20260614T1217Z-0e999dc/runs/h120resumable-d256-l12-loop6-s14300-20260614T1217Z-0e999dc/checkpoints/latest.pt`.
- Checkpoints are not tracked by Git.

## Results

The run was stopped after step9800 quality eval and step10000 train checkpoint because the GPU1 lease had about 22 minutes remaining, not enough to safely reach step10800 eval plus archive metadata.

| checkpoint | h96 loop6 exact | h108 loop6 exact | h120 loop3 exact | h120 loop6 exact | h120 loop6 blank_acc | h132 loop6 exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6800 | 0.9941 | 0.8203 | 0.0039 | 0.0137 | 0.5658 | 0.0000 |
| 7800 | 0.9941 | 0.8926 | 0.0156 | 0.0332 | 0.6062 | 0.0000 |
| 8800 | 0.9922 | 0.9004 | 0.0254 | 0.0410 | 0.6382 | 0.0000 |
| 9800 | 0.9961 | 0.9062 | 0.0312 | 0.0586 | 0.6655 | 0.0000 |

## Insight

The first four quality checkpoints exactly reproduce the previous clean h120 curve: `0.0137 -> 0.0332 -> 0.0410 -> 0.0586`. This makes the positive clean-compute slope solid, but it also makes the shallow slope solid.

The middle foundation is not broken. h96 stays around `0.99`, and h108 rises to `0.9062`. The cliff is specifically the 108-to-120 jump.

Loop still matters at h120: at step9800, h120 exact moves from loop1 `0.0` to loop3 `0.0312` to loop6 `0.0586`. But the loop gain is not large enough to make h120 a supported regime, and h132 remains fully closed.

Resumable training is now proven useful. The remote has `latest.pt` at step10000, so the next clean-compute experiment can continue from this run instead of replaying the first 3 hours.

## Decision

Do not run another identical from-scratch D256 h120 extension. If continuing clean scaling, resume from `latest.pt` and target later checkpoints directly. Otherwise, the next mechanism should be a very simple FutureSeed/loop state update that improves late correction. Do not pivot to selector work, Sudoku repair, scratch noise/Gaussian sweeps, deeper loop-count repeats, or non-resumable D320 repeats.
