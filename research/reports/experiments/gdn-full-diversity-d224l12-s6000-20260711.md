# GDN Full-Diversity D224/L12 Scale Test

## 1. Metainfo

- Plan ID: `P-SCALE-029`
- Status: preparing data
- Planned: 2026-07-11 22:43 CST / 2026-07-11T14:43:00Z
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Source SHA: pending clean commit
- Run: pending

## 2. Hypothesis

The current official-data runs do not train on one million independent Sudoku
problems. The released `sudoku-extreme-1k-aug-1000` train split contains exactly
1,000 source groups and 1,001 symmetry variants per group. The model therefore
sees many masks and legal transformations, but only 1,000 underlying source
problems.

Sudoku-Extreme provides 3,831,994 independent raw training problems. If hard
full-board exactness is limited by relational data diversity rather than another
small state or loss mechanism, replacing the augmented 1k split with the full
raw split should steepen the D224 learning curve without changing the model.

This is a direct bitter-lesson test: scale independent data while holding the
backbone, recurrent compute, optimizer, curriculum, and evaluation fixed.

## 3. Configuration

- Data variable only:
  - old: 1,000 source boards, 1,001 variants each, 1,001,000 rows;
  - new: 3,831,994 raw independent train boards, no Sudoku-specific augmentation.
- Evaluation: unchanged official test split with 422,786 rows.
- Model: native FutureSeed GDN, D224/L12/H14/head-dim16, expand-v4.
- Recurrent compute: loop5, every-loop CE, fixed FutureSeed and loop update.
- Batch: microbatch32, gradient accumulation4, effective batch128.
- Curriculum: `46-50:100,51-55:5900`.
- Target: step6000 from scratch.
- Checkpoints: 1000, 3000, 4500, 6000 on holes53/60/64.
- Disabled: noise, scratch, repair, search, selector, oracle rollout, task rules,
  seed/LR/loss/width sweeps.

## 4. Prediction And Kill Criteria

Matched old-data references:

- step1000 holes53 exact/blank: approximately `0.0176/0.5284`;
- step3000 holes53 exact/blank: `0.0352/0.5750`;
- step6000 holes53 exact/blank: `0.1680/0.6120`;
- step6000 official 51-55 exact: `0.2832`;
- step6000 official 56-64 exact: `0.0801`.

Success requires a better generalization curve, not merely lower train CE:

- primary: step6000 holes53 exact `>0.1680` and official 51-55 `>0.2832`;
- hard-range success: official 56-64 `>=0.12` without losing 51-55;
- strong success: holes60 or holes64 checkpoint exact reaches `>=0.25`.

Kill criteria:

- stop on wrong GPU, NaN, OOM, malformed data, or source/checkpoint ambiguity;
- at step3000, stop only if holes53 exact is `<0.02`, blank accuracy is `<0.54`,
  and the 1000-to-3000 curve has no acceleration;
- if full diversity only lowers train loss while matched exact/blank stay flat,
  classify data diversity as insufficient and do not build a dataset-size table.

## 5. Commands

Pending local browser download, upload, deterministic conversion, GPU-only data
loader/kernel validation, and detached launch.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

No tag unless the final mechanism/scaling result is strong and clean.
