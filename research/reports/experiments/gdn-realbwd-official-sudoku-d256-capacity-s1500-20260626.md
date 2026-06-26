# GDN Real-Backward Official Sudoku D256 Capacity Gate

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d256-capacity-s1500-20260626`
- plan ID: `P-GDN-008`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: `6ea0b0c4b85d595b62ee25cd7623b0d94f6c3042`
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-26 15:15-15:46 UTC

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

- AIStation row: GPU1 only
- host: `cv7rrdngf4kml-0`
- GPU: NVIDIA A800-SXM4-80GB
- remote launch worktree: `/huyang2/double-loop/.worktrees/launch-gdn-d256-6ea0b0c-20260626`
- remote run worktree: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d256-fs-capacity-6ea0b0c-20260626`
- Python: `/opt/conda/bin/python`
- torch: `2.7.0+cu126`
- observed GPU memory: about `34.9GB`
- run metadata reports `git_dirty=false`

## 5. Commands

Launched from a clean detached launch worktree because the remote root checkout
contained historical untracked run files:

```bash
SOURCE_SHA=6ea0b0c4b85d595b62ee25cd7623b0d94f6c3042 \
RUN_STAMP=20260626T1515Z \
RUN_NAME=gdn-d256-official-sudoku-fs-capacity-s1500-20260626T1515Z-6ea0b0c \
bash /huyang2/double-loop/.worktrees/launch-gdn-d256-6ea0b0c-20260626/artifacts/launch/gdn_d256_fs_capacity_s1500_20260626.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exactly this run and record as memory boundary.
- Step100 wall time above 15 minutes: kill exact PID and relaunch only if the bottleneck is clearly microbatch memory, using smaller microbatch with the same effective batch.
- Step300 CE worse than the D192/L10 1500 run at comparable time by a large margin, with no loop gain: stop, because capacity is not opening efficiently.

## 6. Artifacts

- run: `runs/gdn-d256-official-sudoku-fs-capacity-s1500-20260626T1515Z-6ea0b0c`
- visual: `runs/gdn-d256-official-sudoku-fs-capacity-s1500-20260626T1515Z-6ea0b0c/visualizations/index.html`
- result JSON: `runs/gdn-d256-official-sudoku-fs-capacity-s1500-20260626T1515Z-6ea0b0c/output/futureseed_loop_seed52.json`
- score: `runs/gdn-d256-official-sudoku-fs-capacity-s1500-20260626T1515Z-6ea0b0c/score.json`
- remote log: `/huyang2/double-loop/artifacts/logs/gdn-d256-official-sudoku-fs-capacity-s1500-20260626T1515Z-6ea0b0c.log`

## 7. Results

Training curve:

| step | CE | loop1 CE | loop5 CE |
|---:|---:|---:|---:|
| 100 | 1.8019 | 1.8017 | 1.8019 |
| 200 | 1.2742 | 1.2820 | 1.2742 |
| 300 | 1.1411 | 1.2183 | 1.1411 |
| 400 | 1.0963 | 1.1997 | 1.0963 |
| 500 | 1.0544 | 1.2028 | 1.0544 |
| 600 | 1.0368 | 1.2015 | 1.0368 |
| 700 | 1.0521 | 1.1909 | 1.0521 |
| 800 | 1.0150 | 1.1813 | 1.0150 |
| 900 | 1.0188 | 1.2336 | 1.0188 |
| 1000 | 1.0143 | 1.2740 | 1.0143 |
| 1100 | 0.9975 | 1.2270 | 0.9975 |
| 1200 | 0.9581 | 1.1371 | 0.9581 |
| 1300 | 0.9869 | 1.2561 | 0.9869 |
| 1400 | 1.0355 | 1.2402 | 1.0355 |
| 1500 | 0.9991 | 1.2564 | 0.9991 |

Full-board eval, official test split, `eval_n=2048`:

| loop | exact | blank_acc | early | late | early-late gap |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0020 | 0.4481 | 0.4575 | 0.4454 | +0.0121 |
| 2 | 0.0288 | 0.5187 | 0.5158 | 0.5231 | -0.0073 |
| 3 | 0.0288 | 0.5288 | 0.5275 | 0.5320 | -0.0046 |
| 4 | 0.0288 | 0.5294 | 0.5283 | 0.5319 | -0.0036 |
| 5 | 0.0288 | 0.5300 | 0.5291 | 0.5329 | -0.0038 |

Comparisons against D192/L10 GDN+FutureSeed 1500-step:

- exact: `0.0293 -> 0.0288`, slight regression.
- blank_acc: `0.5337 -> 0.5300`, slight regression.
- loop1->loop5 exact gain: `+0.0239 -> +0.0269`, slightly stronger because loop1 is weaker.
- loop1->loop5 blank_acc gain: `+0.0790 -> +0.0819`, slightly stronger, but final blank_acc is lower.
- train time: `2104.5s`; D256 is substantially slower than D192.
- FS state is active: `fs_gate=0.489`, `fs_state_norm=7.828`.

## 8. Conclusions

Decision: mark P-GDN-008 done as a negative capacity boundary. Do not continue blind D256/L12 or wider GDN scaling under the same data/objective.

This run answered the intended question. D256/L10 is runnable on GPU1 with microbatch64 and grad accumulation, so memory is not the blocker. But it does not beat D192/L10 at the same effective batch and step budget. Training CE is noisy and mostly tracks D192 rather than clearly improving it; final exact and blank accuracy are both slightly worse.

The loop still adds useful refinement, but it does not become a stronger final solver just because hidden/state width increased. That matters for the scaling story: FutureSeed+GDN is real, but the current high-ROI axis is no longer simple width. The next better-lesson move should be more data/curriculum, longer training only if paired with a reason to reduce CE noise, or a simple generic FutureSeed/loop state dynamics change. It should not be another no-FS repeat or D256/D320 table.

No tag: primary score is far below `0.50`.

## 9. Submission Record

None.
