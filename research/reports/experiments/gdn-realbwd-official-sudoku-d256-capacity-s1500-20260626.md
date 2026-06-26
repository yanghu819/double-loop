# GDN Real-Backward Official Sudoku D256 Capacity Gate

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d256-capacity-s1500-20260626`
- plan ID: `P-GDN-008`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: pending launch
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-26 UTC

## 2. Hypothesis

P-GDN-007 showed that D192/L10 GDN+FutureSeed has real longer-training slope and reaches the historical RWKV D192/L10 1500-step exact score. The next high-information question is whether more GDN capacity steepens this slope, or whether the current limit is not width/state capacity.

Prediction: if the remaining bottleneck is capacity/state size, D256/L10 should beat D192/L10 at the same optimizer-step budget and effective batch, especially on blank accuracy and loop1-to-loop5 gain. If it only matches D192 while using more memory/compute, the next step should change data/curriculum or FutureSeed state dynamics rather than keep widening.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=256`, `LAYERS=10`, `HEADS=16`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=5`
- microbatch: `64`
- grad accumulation: `2`
- effective batch: `128`
- steps: `1500`
- eval_n: `2048`
- dtype: `bfloat16`
- only arm: `FUTURE_SEED_SCALE=1`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep, no no-FS repeat.

## 4. Environment

Pending GPU1 probe.

## 5. Commands

Pending launch at the final source SHA:

```bash
SOURCE_SHA=<sha> RUN_STAMP=<timestamp> bash /huyang2/double-loop/artifacts/launch/start_gdn_d256_fs_capacity_s1500_20260626.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exactly this run and record as memory boundary.
- Step100 wall time above 15 minutes: kill exact PID and relaunch only if the bottleneck is clearly microbatch memory, using smaller microbatch with the same effective batch.
- Step300 CE worse than the D192/L10 1500 run at comparable time by a large margin, with no loop gain: stop, because capacity is not opening efficiently.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

None.
