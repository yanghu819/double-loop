# GDN Real-Backward Official Sudoku Scale Gate

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-scale-20260626`
- plan ID: `P-GDN-005`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: pending launch
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-26 UTC

## 2. Hypothesis

The real Triton backward removes the GDN training bottleneck. The next question is no longer whether the kernel runs; it is whether native FutureSeed transfers beyond RWKV.

If FutureSeed is a generic terminal-state source of cheap future context, then a larger Gated DeltaNet backbone should show better opening/sample-efficiency with FutureSeed than without it on the same official Sudoku data. If the gain is RWKV-specific, matched no-FS and FS GDN should move together.

This is one matched scale gate, not a seed sweep.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0` to isolate the local recurrent state path
- size: `D_MODEL=128`, `LAYERS=6`, `HEADS=8`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=4`
- batch target: `128`; fallback `96` or `64` only for OOM/speed kill
- steps: `600`
- dtype: `bfloat16`
- arms:
  - no-FS: `FUTURE_SEED_SCALE=0`
  - FS: `FUTURE_SEED_SCALE=1`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep.

## 4. Environment

Pending GPU1 restart/probe.

## 5. Commands

Pending launch script under `/huyang2/double-loop/artifacts/launch/`.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

None.
