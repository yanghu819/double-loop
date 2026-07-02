# GDN Transition Mixed Hard D224/L12 Step6300

## 1. Metainfo

- run_name: `gdn-transition-mixedhard-d224l12-step6300-20260702`
- plan_id: `P-DIAG-017`
- machine: AIStation `GPU1` only
- local_start_time: `2026-07-02 22:30 CST`
- status: `in-progress`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

D224/L12 step6000 is the best current clean point. Pure `56-64` tail did not improve, but that might be because the continuation distribution was too narrow. A mixed `51-64` hard distribution is a generic data-scaling test: keep the transition support while exposing harder boards.

Prediction: if the issue is data distribution rather than state formulation, a short step6000->6300 continuation should improve holes60 / official `56-64` without erasing holes53 / `51-55`.

## 3. Configuration

- Resume checkpoint: `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/checkpoints/train_state_step006000.pt`
- Data: official EqR Sudoku arrays
- Backbone: native FutureSeed GDN, D224/L12/H14/D16, `GDN_EXPAND_V=4.0`
- Loop: loop5, `LOOP_LOSS=all`
- Continuation distribution: `HOLES_MIN=51`, `HOLES_MAX=64`
- Resume quirk: set `FULL_STEPS=6300`, `EVAL_CHECKPOINT_STEPS=6300`; run only until global step6300 and stop.

## 4. Environment

Pending remote launch.

## 5. Commands

Planned command uses `/opt/conda/bin/python`, `CUDA_VISIBLE_DEVICES=0`, no CPU smoke, no GPU2, no scratch, no selector/search/repair.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

No submission/tag planned.
