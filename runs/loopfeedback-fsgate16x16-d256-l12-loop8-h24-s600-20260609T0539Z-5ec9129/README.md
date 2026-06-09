# Loop Feedback + Learned FutureSeed Gate Probe

Run: `loopfeedback-fsgate16x16-d256-l12-loop8-h24-s600-20260609T0539Z-5ec9129`

Source SHA: `5ec9129db2ffe486f388b25f10be491097a1af38`

GPU: GPU1 only, NVIDIA A800-SXM4-80GB.

## Question

Can the clean FutureSeed/RWKV loop make later recurrent loops useful on 16x16 if the next loop explicitly sees the previous loop's prediction distribution, and if FutureSeed learns how much cross-layer state to carry?

This is not a table-filling ablation. It tests one mechanism-level claim: if the loop is neutral because the next iteration lacks a stable belief to refine, generic prediction feedback should create a measurable loop gain.

## Configuration

- Board: 16x16 Sudoku, random holes, 4x4 boxes.
- Model: D256, 12 layers, 8 heads, head dim 32, channel mult 4.
- Loop: max loops 8, L-cycles 2.
- Curriculum: `8-16:300,16-24:300`.
- Batch/eval: batch 16, eval N 256.
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_DECAY=0.5`, `FUTURE_SEED_UPDATE=learned`.
- Loop feedback: `LOOP_FEEDBACK_SCALE=1`.
- Loop supervision: `LOOP_LOSS=shaped`, weights biased toward later loops.
- Kernel: RWKV CUDA path.

## Result

Training ran in `973.0s`, fit at about 43GB, and used GPU1 heavily.

| readout | value |
|---|---:|
| step100 CE | 1.1880 |
| step300 CE | 1.0158 |
| step600 CE | 0.9810 |
| h16 loop8 blank_acc | 0.3501 |
| h24 loop1 blank_acc | 0.3249 |
| h24 loop8 blank_acc | 0.3270 |
| h24 loop8 exact | 0.0000 |
| h32 loop8 blank_acc | 0.2990 |
| loop8 - loop1 blank_acc on h24 | +0.0021 |
| learned FutureSeed update mean | 0.543 |
| loop feedback norm | 1.541 |

## Decision

Discard as a 16x16 rescue direction.

The important point is that this did not fail because the new path was unused. The feedback vector reached nonzero norm and the FutureSeed update gate moved to a learned value around `0.543`. The mechanism was active, but it did not raise per-blank accuracy and did not make loop depth materially useful.

## Insight

The clean 16x16 ceiling is still local digit reliability, not missing explicit loop feedback. At h24, blank accuracy stayed near `0.33`, below the previous fixed-decay/all-loop probes. Exact score is expected to be zero at that local accuracy.

This pushes the next high-ROI work away from feedback/gate micro-tuning. If we keep the clean Bitter Lesson line, the next mechanism needs to improve representation capacity or training credit assignment enough to raise per-cell accuracy first. If we allow structure again, prior unit-memory results already show the direction that works, but that is a separate, less clean path.

## Artifacts

- `config.json`
- `score.json`
- `metadata.json`
- `logs/run.log`
- `launcher.log`
- `output/futureseed_loop_seed52.json`
- `output/futureseed_loop_seed52.md`
- `output/futureseed_loop_case_seed52.html`
- `source_HEAD.txt`
- `source.patch`
