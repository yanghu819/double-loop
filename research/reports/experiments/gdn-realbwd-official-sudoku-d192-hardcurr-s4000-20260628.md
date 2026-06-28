# GDN Real-Backward Official Sudoku D192 Hard Curriculum Scaling

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-hardcurr-s4000-20260628`
- plan ID: `P-GDN-011`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- local branch: `codex/gpu1-experiment-tracking`
- run time: scheduled 2026-06-28 UTC

## 2. Hypothesis

P-GDN-010 showed that same-config D192/L10 GDN+FutureSeed 1500->3000 training improves CE and blank accuracy but barely moves full-board exact. The official train split is already large and hard (`46-64` blank cells), but the runner previously ignored `HOLE_STAGES` in official-data mode.

Mechanism question: is the exact plateau partly caused by weak hard-stage pressure, where the model keeps learning average blank accuracy but does not spend enough budget on the hardest missing-token regime?

Prediction: if hard-stage data pressure is the missing better-lesson axis, a clean blank-count curriculum should move loop5 exact above the `~0.03` plateau, ideally `>=0.06`, without adding rules or repair. If only CE/blank accuracy improve while exact stays around `0.03`, then data hardness alone is not the current bottleneck.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- data change: official loader now supports blank-count filtering through `HOLE_STAGES`
- curriculum: `46-55:800,56-64:3200`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=5`
- batch: `128`
- steps: `4000`
- eval_n: `2048`
- checkpoint eval: `1500,2500`, no train checkpoint save
- dtype: `bfloat16`
- optimizer: `LR=0.0015`, `WEIGHT_DECAY=0.001`
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_UPDATE=fixed`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep, no gate/decay/loss-weight table.

## 4. Environment

- host: AIStation `GPU1`
- GPU target: A800/A100 80GB class GPU, `CUDA_VISIBLE_DEVICES=0`
- remote run worktree: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-hardcurr-s4000-<sha>-20260628`
- git dirty at run start: must be `false` after detached checkout

## 5. Commands

Launch from GitHub-truth SHA:

```bash
SOURCE_SHA=<sha> \
RUN_STAMP=<timestamp> \
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_hardcurr_s4000_20260628.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exact PID and record as failed infrastructure.
- Step100 wall time above 15 minutes: stop; curriculum run is too slow for ROI.
- Checkpoint step1500 has CE much worse than the 3000-step clean run and loop5 exact still zero: stop as optimization failure.
- Checkpoint step2500 shows blank accuracy improving but exact flat near the old plateau: continue only if CE slope is still strong; otherwise stop and archive as data-curriculum boundary.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

None.
