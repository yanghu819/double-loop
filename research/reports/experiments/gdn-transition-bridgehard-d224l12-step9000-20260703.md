# GDN Transition Bridge Hard D224/L12 Step9000

## 1. Metainfo

- run_name: `gdn-transition-bridgehard-d224l12-step9000-20260703T0600Z-<sha>`
- plan_id: `P-DIAG-019`
- machine: AIStation `GPU1` only
- local_start_time: `2026-07-03 14:00 CST`
- status: `planned`
- branch: `codex/gpu1-experiment-tracking`

## 2. Hypothesis

P-DIAG-018 showed that broad mixed `51-64` training is better than a narrow pure `56-64` tail, but the hardest official range remains weak. The working hypothesis is that `56-64` boards need a continuous bridge from the already-learned `51-55` transition. If the model is pushed directly into hardest-only data, it loses useful support; if it sees a staged bridge, it may preserve transition skill while shifting the boundary upward.

This is still a bitter-lesson experiment: more data distribution, more compute, same generic model. No Sudoku repair, no selector, no handwritten rule.

## 3. Configuration

- Resume checkpoint: P-DIAG-018 `train_state_step007000.pt`.
- Data: official EqR Sudoku arrays.
- Backbone: native FutureSeed GDN, D224/L12/H14/D16, `GDN_EXPAND_V=4.0`.
- Loop: loop5, `LOOP_LOSS=all`.
- Curriculum:
  - historical skipped prefix: `46-50:100,51-55:5900,51-64:1000`
  - active continuation: `51-55:600,51-64:800,56-64:600`
- Target: `FULL_STEPS=9000`, checkpoint eval at `7600,8400,9000`.
- No scratch, no extra loss, no repair/search/selector, no CPU smoke, no GPU2.

## 4. Prediction And Kill Criteria

Prediction:

- Success if official `56-64` loop5 exact moves from P-DIAG-018 `0.0859` to `>=0.12`, while official `51-55` remains `>=0.25`.
- Stronger success if holes60 loop5 exact reaches `>=0.23` without blank accuracy dropping.

Kill criteria:

- If step7600 holes53 collapses far below P-DIAG-018 final `0.1914/0.6568`, the bridge stage is harming transition skill.
- If step8400 holes60 does not beat P-DIAG-018 best `0.2031/0.6666` and CE/blank has no slope, stop instead of burning a same-shape 9000 repeat.
- Stop immediately on NaN, OOM, or GPU utilization anomaly.

## 5. Commands

Launch command will be archived in the run directory as `launch.env` and `launch.sh`.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

No tag unless the score is meaningfully strong and the mechanism conclusion remains clean.
