# EqR FutureSeed Mechanism Study - 2026-06-04

Source SHA: `0cc0d1bedf69e1f0aef8fa7a3ab3d4a61e0666e1`  
GPU: AIStation GPU1 only, `CUDA_VISIBLE_DEVICES=0`, no CPU smoke, no Hydra, no flash-attn, feature noise disabled.

Provenance caveat: the first 8 loop-budget runs were `git_dirty=false`. The later 12 runs were `git_dirty=true` only because the first batch had already appended metadata rows to `leaderboard.csv`; inspected `source.patch` shows leaderboard metadata diffs, not code or experiment-config changes.

## Hypothesis

FutureSeed may matter in EqR only when recurrent compute or representation capacity is constrained. We therefore tested loop-budget compression and capacity bottleneck instead of repeating saturated random-hole runs.

## Aggregate Results

- loop_budget loop2 base: n=6, train_sec=300.3, h16=0.9964, h20=0.9827, h24=0.9111, h28=0.7041, h32=0.2946
- loop_budget loop2 fs_only: n=6, train_sec=301.1, h16=0.9974, h20=0.9860, h24=0.9232, h28=0.7113, h32=0.3171
- bottleneck 96x1 loop4 base: n=4, train_sec=267.4, h16=0.0586, h20=0.0083, h24=0.0000, h28=0.0000, h32=0.0000
- bottleneck 96x1 loop4 fs_only: n=4, train_sec=272.1, h16=0.0532, h20=0.0054, h24=0.0000, h28=0.0000, h32=0.0000

## Paired Deltas

### loop_budget
- h24: mean +0.0120; seed52:-0.0039, seed53:+0.0117, seed54:+0.0254, seed55:-0.0020, seed56:+0.0059, seed57:+0.0352
- h28: mean +0.0072; seed52:-0.0332, seed53:+0.0508, seed54:+0.0352, seed55:-0.0020, seed56:-0.0020, seed57:-0.0059
- h32: mean +0.0225; seed52:-0.0430, seed53:+0.0625, seed54:+0.0801, seed55:-0.0020, seed56:-0.0078, seed57:+0.0449

### bottleneck
- h24: mean +0.0000; seed52:+0.0000, seed53:+0.0000, seed54:+0.0000, seed55:+0.0000
- h28: mean +0.0000; seed52:+0.0000, seed53:+0.0000, seed54:+0.0000, seed55:+0.0000
- h32: mean +0.0000; seed52:+0.0000, seed53:+0.0000, seed54:+0.0000, seed55:+0.0000

## Decision

- Loop-budget compression: weak/inconsistent tail lift only. h32 mean delta is below +0.05 and sign flips across seeds, so this is not a solid FutureSeed win.
- Capacity bottleneck: FutureSeed does not act as a useful bottleneck prior in EqR. h28/h32 are not improved under 96 hidden / 1 layer; low exact is a capacity/global-coupling failure, not feature-noise absence.
- Structured OOD eval was skipped by the stated rule because neither experiment produced a clear positive FutureSeed signal.
- Next high-ROI direction is not EqR+FutureSeed tuning. Either move FutureSeed back to RWKV7/CUDA where it matched the recurrent dynamics, or change EqR injection mechanism rather than scaling more seeds.

## Lessons Backed Up

- FutureSeed can look useful on individual hard-tail seeds, but seed-paired means are the only defensible readout here.
- Compressing loop count gave a much better mechanism test than saturated 9x9 random holes: base remained strong, exposing that FutureSeed is at best a small tail perturbation.
- Compressing capacity created the desired hard regime, but FutureSeed did not rescue it; this argues against treating FutureSeed as a generic representation prior for EqR.
- Feature-diff noise should stay stopped until there is a new selector/trajectory hypothesis; as training noise it is too expensive for the observed information gain.

## Files

- `runs.csv`: per-run metrics and config summary.
- `summary.json`: machine-readable aggregate and decision notes.
