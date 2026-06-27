# GDN Real-Backward Official Sudoku D192 Loop-Residual FutureSeed

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-loopresid-s1500-20260627`
- plan ID: `P-GDN-009`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: `629b5787b2c579a487f51d9d9aa776248b66ab16`
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-27 03:28-03:43 UTC

## 2. Hypothesis

P-GDN-008 showed that simple D192->D256 widening does not improve official Sudoku quality. The next high-ROI question is whether the bottleneck is loop state dynamics: fixed FutureSeed recomputes seed state each loop, while `loop_residual` lets the model carry a damped FutureSeed memory across loops and learn how much to update it.

Prediction: if later loops are limited because FutureSeed state cannot persistently revise itself, loop-residual seeding should improve loop5 exact or blank accuracy over fixed D192/L10, or at least make loop3-5 continue improving instead of saturating at loop2. If it matches or regresses, the next axis should be data/curriculum rather than more state-memory plumbing.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=5`
- batch: `128`
- steps: `1500`
- eval_n: `2048`
- dtype: `bfloat16`
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_UPDATE=loop_residual`, `FUTURE_SEED_DECAY=0.5`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep, no no-FS repeat.

## 4. Environment

- host: AIStation `GPU1`
- GPU: NVIDIA A100-SXM4-80GB
- device: CUDA only; no CPU smoke
- torch: `2.7.0+cu126`
- remote run worktree: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-loopresid-629b578-20260627`
- launch worktree: `/huyang2/double-loop/.worktrees/launch-gdn-loopresid-629b578-20260627`
- git dirty at run start: `false`

## 5. Commands

Launched from detached GitHub-truth SHA:

```bash
SOURCE_SHA=629b5787b2c579a487f51d9d9aa776248b66ab16 \
RUN_STAMP=20260627T0135Z \
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_loopresid_s1500_20260627.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exactly this run and record as failed mechanism gate.
- Step100 wall time above 15 minutes: stop; this mechanism adds too much overhead.
- Step300 CE much worse than fixed D192/L10 without compensating loop gain: stop as optimization failure, not a score result.

## 6. Artifacts

- local run: `runs/gdn-d192-official-sudoku-fs-loopresid-s1500-20260627T0135Z-629b578`
- visual dashboard: `runs/gdn-d192-official-sudoku-fs-loopresid-s1500-20260627T0135Z-629b578/visualizations/index.html`
- case HTML: `runs/gdn-d192-official-sudoku-fs-loopresid-s1500-20260627T0135Z-629b578/output/futureseed_loop_case_seed52.html`
- result JSON: `runs/gdn-d192-official-sudoku-fs-loopresid-s1500-20260627T0135Z-629b578/output/futureseed_loop_seed52.json`
- result MD: `runs/gdn-d192-official-sudoku-fs-loopresid-s1500-20260627T0135Z-629b578/output/futureseed_loop_seed52.md`
- score: `runs/gdn-d192-official-sudoku-fs-loopresid-s1500-20260627T0135Z-629b578/score.json`
- remote log: `/huyang2/double-loop/artifacts/logs/gdn-d192-official-sudoku-fs-loopresid-s1500-20260627T0135Z-629b578.log`

## 7. Results

Primary score: `metrics.eval_clean.loop5.label_exact = 0.029296875`.

Training curve:

| step | CE | loop1 CE | loop-last CE |
|---:|---:|---:|---:|
| 100 | 1.7984 | 1.7983 | 1.7984 |
| 300 | 1.1194 | 1.1632 | 1.1194 |
| 600 | 1.0341 | 1.1621 | 1.0341 |
| 900 | 1.0236 | 1.1506 | 1.0236 |
| 1200 | 0.9348 | 1.1308 | 0.9348 |
| 1500 | 0.9901 | 1.2194 | 0.9901 |

Official Sudoku test eval, `eval_n=2048`:

| loop | exact | blank_acc | early | late | fs_mem | fs_mem_delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0078 | 0.4520 | 0.4487 | 0.4500 | 5.296 | 0.000 |
| 2 | 0.0293 | 0.5193 | 0.5163 | 0.5218 | 5.077 | 1.548 |
| 3 | 0.0293 | 0.5333 | 0.5283 | 0.5352 | 4.995 | 0.954 |
| 4 | 0.0293 | 0.5342 | 0.5298 | 0.5372 | 4.968 | 0.526 |
| 5 | 0.0293 | 0.5344 | 0.5301 | 0.5372 | 4.959 | 0.280 |

Comparison against fixed D192/L10 GDN+FutureSeed 1500-step:

- fixed D192 loop5 exact: `0.029296875`; loop-residual: `0.029296875`。
- fixed D192 loop5 blank_acc: `0.5337`; loop-residual: `0.5344`。
- fixed D192 step1500 CE: `0.9875`; loop-residual: `0.9901`。
- loop-residual loop1->loop5 exact gain: `+0.0215`; fixed D192 was `+0.0239`。
- loop-residual loop1->loop5 blank_acc gain: `+0.0824`; fixed D192 was `+0.0790`。

## 8. Conclusions

Decision: mark P-GDN-009 as discarded/negative for the current bottleneck.

The mechanism is alive but not decisive. `loop_residual` makes the FutureSeed memory change across loops (`fs_mem_delta` nonzero) and later loops improve blank accuracy, but exact saturates at loop2 and the final loop5 exact ties the fixed D192 baseline. The final CE is also not better.

The insight is useful: the current limit is not simply that FutureSeed state cannot persist across loops. A damped residual memory can move soft cell accuracy but does not create stronger global Sudoku consistency. This pushes the next high-ROI direction away from more small state-update variants and toward a more bitter-lesson axis: more data/curriculum or a larger training budget with the already-clean GDN+FutureSeed backbone, unless we introduce a genuinely generic training signal that targets global consistency without Sudoku rules.

Do not sweep `FUTURE_SEED_DECAY`, update gate, seed, or another fixed residual variant. That would be table filling.

## 9. Submission Record

None.
