# FutureSeed Sudoku Official RWKV7 Four-Backbone Rerun

## 1. Metainfo

- Experiment: `E-BASELINE-003`
- Plan: `P-BASELINE-003`
- Status: in progress
- Scheduled: 2026-07-24
- Machine: AIStation GPU1, NVIDIA A800-SXM4-80GB
- GPU2: forbidden
- CPU model smoke: forbidden
- Source: the final detached SHA will be recorded by each run's `config.json`

## 2. Hypothesis

The previous public `RWKV` arm used a local RWKV-style frontend rather than
official RWKV7 TimeMix. Its lower score therefore cannot support a GDN versus
RWKV7 conclusion.

If the finite-budget carrier ordering is real, official RWKV7 TimeMix should
still trail GDN on hard opening under the same data, recurrent-state size,
FutureSeed rule, loop budget, loss, optimizer contract, and evaluator. If the
ordering flips, the old conclusion must be withdrawn.

This experiment compares recurrent token mixers inside one shared research
shell. It does not claim to reproduce the complete RWKV language model.

## 3. Configuration

All four arms use:

- public carriers: RWKV7 TimeMix, official FLA GDN, official FLA GDN2, and
  official FLA KDA;
- D192, 10 layers, 6 heads, head dimension 32;
- matched recurrent state shape `6 x 32 x 32`;
- native unit-normalized FutureSeed, scale 1, fixed update;
- five loops and equal cross-entropy supervision on every loop;
- microbatch 32, accumulation 4, effective batch 128;
- BF16, AdamW, LR 0.0015, weight decay 0.001, seed 52;
- shared grouped weight decay: matrix `.weight` parameters decay, normalization
  and explicit no-weight-decay parameters do not;
- official full-diversity Sudoku, curriculum `46-50:100,51-55:400`;
- 512-board mixed and official blank-range evaluation;
- a strict 512-board 53-blank checkpoint evaluation, rather than a mislabeled
  sample from the full test distribution;
- no noise, feedback, scratch state, selector, repair, search, or task rule.

Parameter count, wall time, and VRAM are measured rather than padded.

RWKV7 provenance is pinned to BlinkDL/RWKV-LM commit
`952102498e9ed367ea0a59ee64106916d474d30f`, model blob
`b4d167fedead2655d253c55eb47b65f00e7193d2`, and clamp-kernel blob
`827faeb06b9d2b6e31b3efe85af6d3ae4cf88905`.

## 4. Environment

- Remote project root: `/huyang2/double-loop`
- Remote run worktree: detached exact GitHub SHA under
  `/huyang2/double-loop/artifacts/worktrees/`
- Python: `/opt/conda/bin/python`
- Cache roots: `/huyang2/double-loop/.cache`
- Data: `/huyang2/double-loop/data/sudoku-extreme-full`
- Models: `/huyang2/double-loop/models`
- Runs: `/huyang2/double-loop/runs`
- Required environment: `CUDA_VISIBLE_DEVICES=0`,
  `FLA_DISABLE_BACKEND_DISPATCH=1`, `FLA_CONV_BACKEND=triton`
- Formal launch also requires a clean source tree and writes artifacts through
  `RUNS_ROOT=/huyang2/double-loop/runs`.

## 5. Commands

Preflight:

```bash
CUDA_VISIBLE_DEVICES=0 \
PERSIST_ROOT=/huyang2/double-loop \
PYTHON_BIN=/opt/conda/bin/python \
./scripts/run_sudoku_baseline_preflight.sh <preflight-dir>
```

Formal suite, only after preflight passes:

```bash
CUDA_VISIBLE_DEVICES=0 \
PERSIST_ROOT=/huyang2/double-loop \
PYTHON_BIN=/opt/conda/bin/python \
BENCHMARK_STEPS=500 \
SUITE_ID=<timestamp>-<sha> \
./scripts/run_sudoku_backbone_suite.sh
```

Hard stop conditions:

- any source, formula, fused-decay, CUDA/Torch forward/state/backward,
  state-continuity, `v_first`, optimizer, official-FLA, data, evaluator, or
  no-fallback gate fails;
- NaN, OOM, wrong GPU, or low utilization with high memory;
- one arm takes more than 15 minutes to reach step 100.

There is no automatic kernel, batch, device, or implementation fallback.

## 6. Artifacts

The first RWKV7 launch at SHA `7307637` was rejected before optimizer step 1:
its `config.json` reported `git_dirty=true`. Review traced the dirtiness to
eight untracked ELF core files emitted by the `ninja` compiler while building
CUDA extensions, not to the model process. The exact launch PIDs were stopped,
the core files and hashes were moved to
`/huyang2/double-loop/artifacts/core-dumps/`, and the non-result was archived
with `abort.json` under `/huyang2/double-loop/runs/`.

The relaunch contract now fails closed on any source dirtiness, disables core
dumps for preflight/formal compilation, and keeps all run artifacts outside
the detached source worktree.

Every accepted arm will archive `config.json`, `score.json`, logs, source SHA,
source snapshot, checkpoint metadata, official blank-range metrics, and
loop-by-loop same-puzzle visualizations. Model checkpoints remain outside Git.

## 7. Results

Pending strict GPU1 preflight and formal suite.

Review note: before formal training, the checkpoint evaluator was corrected so
that `holes53` really selects exactly 53 blanks on official data. The previous
behavior ignored the requested count only for checkpoint evaluation; final
official blank-range evaluation was already correct. No result from the
mislabelled checkpoint path is used.

Review note: the aborted dirty-tree launch completed zero optimizer steps and
is not a benchmark arm. Its only admissible output is the infrastructure
failure record.

## 8. Conclusions

Pending. A carrier decision requires either at least `+0.03` hard exact or
equal quality with at least 20% lower measured compute. Otherwise the result is
reported only as a one-seed, finite-budget observation.

## 9. Submission Record

No tag or submission is allowed unless the primary score is at least 0.50 and
the mechanism conclusion is clean.
