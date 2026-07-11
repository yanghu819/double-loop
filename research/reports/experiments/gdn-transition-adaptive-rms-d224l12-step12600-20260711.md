# GDN FutureSeed Adaptive-RMS Continuation

## 1. Metainfo

- Plan ID: `P-DIAG-027`
- Status: in-progress
- Start: 2026-07-11 11:06 CST / 2026-07-11T03:06:00Z
- Machine: AIStation `GPU1` only
- Branch: `codex/gpu1-experiment-tracking`
- Source SHA: `e6a3f240ea21fe8d4e6d162d1d3df43a5e7604f8`
- Run: `gdn-transition-adaptiverms-d224l12-step12600-20260711T025509Z-e6a3f24`

## 2. Hypothesis

Native FutureSeed currently normalizes every previous-layer terminal recurrent
state to unit RMS before using it as the next layer's initial state. This keeps
the recurrence numerically stable, but it deletes one generic signal: how large
the raw state was. That magnitude can encode accumulated evidence or uncertainty
without relying on any Sudoku rule.

The experiment asks one question: is discarded state magnitude the missing
information that prevents the hard 53/60/64-blank regime from becoming stable?

The mechanism keeps the normalized state direction unchanged and learns only a
bounded, sample-dependent amplitude from `log(raw_state_rms)`. Its parameters
start at zero, so the initial forward pass is exactly the existing unit-RMS
FutureSeed. This isolates information content from width, depth, batch size,
extra loss, or a new solver.

## 3. Configuration

- Base checkpoint: P-DIAG-021 D224/L12 step12000 train checkpoint.
- Data: official EqR Sudoku arrays, train split `train`, eval split `test`.
- Backbone: native FutureSeed GDN, Triton recurrent forward/backward.
- Model: D224/L12/H14/head-dim16, `GDN_EXPAND_V=4.0`, no short convolution.
- Recurrent compute: loop5, `L_CYCLES=2`, all-loop CE.
- Batch: microbatch32, grad accumulation4, effective batch128.
- FutureSeed update: fixed; loop update: fixed.
- New mechanism: `FUTURE_SEED_NORM_MODE=adaptive_rms`.
- Adaptive gain: `exp(0.5 * tanh(slope * log(raw_rms) + bias))`, bounded to
  approximately `[0.61, 1.65]`, zero initialized at gain `1.0`.
- Continuation: global step12000 to step12600.
- Checkpoints: step12300 and step12600 on holes53/60/64.
- Final eval: official blank ranges 46-50, 51-55, 56-64; case bank loops1/3/5.
- Disabled: feature noise, aggregate noise, scratch, extra loss, repair, search,
  selector, oracle rollout, task-specific rules, seed sweep.

## 4. Prediction And Kill Criteria

Prediction:

- If raw state magnitude carries useful board-dependent confidence, adaptive
  gain standard deviation should become at least `0.01` instead of behaving as
  one global scalar.
- That use must translate to hard quality: holes60 or holes64 loop5 exact should
  reach at least `0.25`, or official 56-64 exact should reach at least `0.17`
  while official 51-55 remains at least `0.33`.
- Loop1 to loop5 exact gain must remain clearly positive.

Kill criteria:

- Stop on NaN, OOM, wrong GPU, or missing checkpoint provenance.
- Stop at step12300 if gain std remains below `0.002`, gain saturates near its
  bound, or both hard exact and blank accuracy clearly leave the P-DIAG-021
  quality regime.
- Discard after step12600 if the gain moves but hard exact does not improve.
- Do not sweep gain range, initialization, learning rate, or seed after failure.

CUDA-only zero-initialization gate on GPU1 A800 passed before launch:

- output max absolute difference, unit vs adaptive: `0.0`
- common-parameter gradient max absolute difference: `0.0`
- adaptive gain means by loop: `1.0, 1.0`
- adaptive sample gain standard deviations by loop: `0.0, 0.0`
- expected new parameters only: `future_seed_norm_slope/bias`

## 5. Commands

Detached worktree:

`/huyang2/double-loop/.worktrees/pdiag027-adaptiverms-e6a3f24-20260711T025509Z`

Launch script:

`/huyang2/double-loop/artifacts/launch/pdiag027/pdiag027_launch.sh`

The script verifies detached SHA and checkpoint existence, then launches the
configuration in section 3 with `CUDA_VISIBLE_DEVICES=0`, `SMOKE_DONE=1`,
`SKIP_SETUP=1`, `SOURCE_SNAPSHOT_MODE=lean`, and
`PYTHON_BIN=/opt/conda/bin/python`. No CPU model smoke is used.

## 6. Artifacts

- Remote worktree:
  `/huyang2/double-loop/.worktrees/pdiag027-adaptiverms-e6a3f24-20260711T025509Z`
- Expected remote run directory:
  `/huyang2/double-loop/.worktrees/pdiag027-adaptiverms-e6a3f24-20260711T025509Z/runs/gdn-transition-adaptiverms-d224l12-step12600-20260711T025509Z-e6a3f24`
- CUDA equivalence script:
  `/huyang2/double-loop/artifacts/launch/pdiag027/pdiag027_equivalence.py`

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

No tag unless the mechanism is both stronger and clean enough for the paper
claim.
