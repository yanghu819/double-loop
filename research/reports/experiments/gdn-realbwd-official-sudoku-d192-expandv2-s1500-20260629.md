# GDN Real-Backward Official Sudoku D192 ExpandV2 State-Capacity Probe

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-expandv2-s1500-20260629`
- plan ID: `P-GDN-014`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: `e2375df572c59dc68b924e996e668946c7b88eb5`
- local branch: `codex/gpu1-experiment-tracking`
- launched: 2026-06-29 04:55 UTC
- recorded: 2026-06-29 05:24 UTC

## 2. Hypothesis

The last several official Sudoku GDN+FutureSeed probes all point to the same bottleneck: FutureSeed opens training, but hard full-board exact stalls around `0.03`. P-GDN-013 showed that simply evaluating/training more loops does not keep fixing hard boards after loop3/5.

Mechanism question: is the fixed D192/L10 GDN state too small to retain enough alternative constraint information for later loops, even though the token width is already adequate to improve blank accuracy?

Prediction: if recurrent value/state capacity is the bottleneck, doubling GDN value/state dimension per head should improve exact and/or visibly reduce hard-case wrong cells without adding task rules. If exact remains around `0.03` while CE/blank accuracy only move softly, then value-state expansion alone is not the missing piece.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- training blank-count range: full official train split, effectively `46-64`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- state/value expansion: `GDN_EXPAND_V=2.0`, per-head value/state dim `32`
- loops: `MAX_LOOPS=5`
- microbatch: `64`, gradient accumulation `2`, effective batch `128`
- steps: `1500`
- eval_n: `2048`
- checkpoint eval: `800,1200`, no train checkpoint save
- case bank: official eval fixed batch, `CASE_BANK_N=3`, `CASE_BANK_EVAL_N=256`, loops `1,2,3,5`
- dtype: `bfloat16`
- optimizer: `LR=0.0015`, `WEIGHT_DECAY=0.001`
- blank CE weight: `8`
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_UPDATE=fixed`
- no exact-margin or extra loss

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule in training, no seed sweep, no loss table.

## 4. Environment

- host: AIStation `GPU1`
- GPU target: A100 80GB, `CUDA_VISIBLE_DEVICES=0`
- remote run worktree: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-expandv2-s1500-<sha>-20260629`
- git dirty at run start: must be `false` after detached checkout

## 5. Commands

Launch from GitHub-truth SHA:

```bash
SOURCE_SHA=e2375df572c59dc68b924e996e668946c7b88eb5 \
RUN_STAMP=20260629T0455Z \
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_expandv2_s1500_20260629.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exact PID and record infrastructure failure.
- Step100 wall time above 12 minutes: expandV2 is too slow for this diagnostic at this batch; relaunch only with lower microbatch if no source change is needed.
- If checkpoint800 has CE above `1.5` and loop5 exact is still zero with blank_acc below `0.45`, stop as optimization failure.
- If checkpoint1200 exact is near baseline and CE/blank show no slope, finish only if final eval is near; otherwise archive as capacity boundary.

Success criteria:

- Strong: loop5 exact `>=0.05`.
- Useful: loop5 exact improves by at least `+0.01` over the clean D192/L10 1500 baseline (`0.0293`), or official hard-case bank shows a clear wrong-cell reduction without just moving errors.
- Negative: exact remains around `0.03` or only CE/blank accuracy improve.

## 6. Artifacts

- run root: `runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df`
- config: `runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df/config.json`
- score: `runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df/score.json`
- log: `runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df/logs/run.log`
- result JSON/MD/HTML: `runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df/output/futureseed_loop_seed52.{json,md,html}`
- checkpoint evals: `runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df/output/checkpoint_eval_step000800.json`, `runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df/output/checkpoint_eval_step001200.json`
- official case-bank visualization: `runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df/output/case_bank/official/index.html`
- visualization hub: `runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df/visualizations/index.html`
- source snapshot: `runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df/source_snapshot.tar.gz`

Preflight note: two initial launches exited before GPU training. The first missed `HOLES_MIN=45 HOLES_MAX=64`; the second used an invalid full SHA. The committed launch fix is `e2375df`, and the recorded run above is the only GPU-consuming run.

## 7. Results

Run completed without abort on GPU1.

Training and runtime:

- final train CE: `0.98117`
- loop1 train loss: `1.16870`
- loop5 train loss: `0.98117`
- train time: `1722.4s`
- max CUDA allocated: `37131 MB`
- max CUDA reserved: `39814 MB`
- final FutureSeed state norm: `10.895`
- final FutureSeed gate mean: `0.481`

Checkpoint slope:

| step | CE | loop5 exact | loop5 blank acc |
|---:|---:|---:|---:|
| 800 | `0.99455` | `0.01758` | `0.51404` |
| 1200 | `0.93593` | `0.01758` | `0.52879` |
| 1500 | `0.98117` | `0.02930` | `0.53637` |

Final loop dynamics on official eval, `eval_n=2048`:

| loop | exact | blank acc |
|---:|---:|---:|
| 1 | `0.00098` | `0.45604` |
| 2 | `0.02539` | `0.51097` |
| 3 | `0.02930` | `0.53122` |
| 4 | `0.02930` | `0.53565` |
| 5 | `0.02930` | `0.53637` |

Compared with the clean D192/L10 GDN+FutureSeed 1500-step baseline, this is not an exact-score win: loop5 exact ties `0.02930`. It does improve soft metrics slightly: blank accuracy is higher than the earlier D192/L10 1500 baseline (`0.5364` vs `0.5337`) and train CE is slightly lower (`0.9812` vs `0.9875`). Compared with P-GDN-013, it also shows better blank accuracy and a lower CE, but the same exact plateau.

Official case-bank, `eval_n=256`, blank counts `46-64`, mean `55.74`:

- final loop5 exact: `0.03516`
- final loop5 blank acc: `0.54103`
- selected visual cases: `3` solved-by-loop and `3` hard failures.

Case trajectories, loops `[1,2,3,5]`:

| case | holes | wrong cells | changed cells | conflict units |
|---|---:|---|---|---|
| `official_solved_by_loop_01_b0089` | 47 | `9,2,0,0` | `0,7,2,0` | `12,4,0,0` |
| `official_solved_by_loop_02_b0131` | 49 | `6,1,0,0` | `0,7,1,0` | `14,3,0,0` |
| `official_solved_by_loop_03_b0095` | 47 | `4,0,0,0` | `0,4,0,0` | `8,0,0,0` |
| `official_hard_failure_01_b0083` | 56 | `23,15,9,5` | `0,24,11,6` | `25,17,15,4` |
| `official_hard_failure_02_b0229` | 55 | `24,17,9,6` | `0,23,11,4` | `27,20,14,11` |
| `official_hard_failure_03_b0178` | 55 | `21,11,10,6` | `0,20,13,4` | `23,19,18,13` |

## 8. Conclusions

This is a weak partial positive for recurrent state capacity, but a negative decision for continuing simple `GDN_EXPAND_V` sweeps.

What worked:

- ExpandV2 makes optimization easier. CE is lower throughout the run, and blank accuracy improves over the fixed-value D192/L10 baseline.
- The hard-case visualizations are slightly more alive than P-GDN-013. Hard failures keep changing through loop5, and final wrong counts such as `5/6/6` are mildly better than the frozen loop8 failures from P-GDN-013.
- This supports the idea that recurrent state capacity is a real axis, not a fake knob.

What failed:

- Full-board exact did not open. Final loop5 exact is exactly `0.029296875`, tying the existing D192/L10 plateau.
- Checkpoint1200 already showed the core mismatch: CE and blank acc improved, while exact stayed flat at `0.01758`.
- More value state alone does not solve the global consistency bottleneck.

Decision:

- Do not run an `expand_v` table (`1.5/2/3`) as the next step.
- Do not claim state capacity alone solves the plateau.
- The next useful experiment should change how loop state is updated or trained, while staying generic: the model needs a mechanism that turns improved per-cell confidence into full-board consistency, not just more room in the state tensor.

Paper implication:

FutureSeed+GDN now has a clean story: FutureSeed opens the model; bigger recurrent state improves soft solving; but exact Sudoku exposes the missing piece. The remaining bottleneck is not obvious compute starvation, it is the state/update rule that should convert local improvements into globally consistent boards.

## 9. Submission Record

None.
