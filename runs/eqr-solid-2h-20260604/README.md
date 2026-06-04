# EqR 2h Solidification Aggregate

## Scope

- GPU1 only, `CUDA_VISIBLE_DEVICES=0`, source commit `66ff740`.
- 24 completed GPU runs: original 2x2 mechanism check plus focused base-vs-combo replications and hard-curriculum replications.
- All recorded configs have `git_dirty=false`; source snapshots are archived per run.

## Decision

Do not mainline feature-diff noise for EqR right now. FutureSeed+feature-diff is viable, but not a solid superiority result over base EqR. Base EqR is already very strong; combo costs about 1.7-2.0x training time and only gives small, high-variance hard-hole deltas.

## Original Curriculum 4-8 -> 8-12

- original_base: n=4, train_sec=207.6, h8=0.9995, h12=0.9868, h16=0.9048, h20=0.6372, h24=0.2354
- original_combo: n=4, train_sec=394.1, h8=0.9990, h12=0.9858, h16=0.9062, h20=0.6421, h24=0.2485
- original_fs_only: n=2, train_sec=215.4, h8=1.0000, h12=0.9854, h16=0.8906, h20=0.5791, h24=0.1865
- original_fd_only: n=2, train_sec=374.1, h8=1.0000, h12=0.9814, h16=0.8926, h20=0.6162, h24=0.2139

## Hard Curriculum 8-16 -> 16-24

- hard_base: n=6, train_sec=299.7, h16=0.9977, h20=0.9886, h24=0.9271, h28=0.7057, h32=0.2803
- hard_combo: n=6, train_sec=530.5, h16=0.9987, h20=0.9824, h24=0.9186, h28=0.7087, h32=0.2939

## Paired Hard Deltas: Combo Minus Base

- seed 52: h16=+0.0000, h20=+0.0059, h24=+0.0000, h28=+0.0684, h32=+0.0898
- seed 53: h16=+0.0020, h20=-0.0117, h24=-0.0293, h28=-0.0410, h32=-0.1133
- seed 54: h16=+0.0000, h20=-0.0059, h24=+0.0273, h28=+0.0977, h32=+0.1250
- seed 55: h16=-0.0020, h20=-0.0059, h24=-0.0234, h28=-0.0625, h32=-0.0762
- seed 56: h16=+0.0020, h20=-0.0039, h24=-0.0059, h28=+0.0195, h32=+0.0918
- seed 57: h16=+0.0039, h20=-0.0156, h24=-0.0195, h28=-0.0645, h32=-0.0352

## Interpretation

- The first single-run positive was real as a viability signal: the combo learns and loop depth matters.
- The stronger 2h evidence changes the claim: EqR base recurrent attractor is already enough for this Sudoku setup; FutureSeed+feature-diff does not robustly dominate it.
- Feature-diff noise is the weak link: it roughly doubles wall time in original runs and does not produce a reliable exact-match lift.
- FutureSeed remains interesting only as a low/no-noise architectural idea, but the current combined FutureSeed+feature-diff recipe should not be the EqR main branch.

## Next High-ROI Step

Stop broad ablation here. If continuing EqR, use base EqR as the control/default and test a cleaner FutureSeed-only variant under a harder distribution or a model-capacity bottleneck. Do not spend more budget on feature-diff noise unless a new hypothesis explains why its cost should buy robustness.
