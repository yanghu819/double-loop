# FutureSeed Sudoku Official RWKV7 Four-Backbone Rerun

## 1. Metainfo

- Experiment: `E-BASELINE-003`
- Plan: `P-BASELINE-003`
- Status: complete
- Scheduled: 2026-07-24
- Machine: AIStation GPU1, NVIDIA A800-SXM4-80GB
- GPU2: forbidden
- CPU model smoke: forbidden
- Source: `6e51f067a18d936bda7f3d588b7b3f32625b2b18`

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
- backbone-independent shared-shell initialization: embedding, positional
  embedding, every shared ChannelMix, and the output head are reset from one
  fixed seed after backbone construction;
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

Every accepted arm archives `config.json`, `score.json`, logs, source SHA,
source snapshot, checkpoint metadata, official blank-range metrics, and
loop-by-loop same-puzzle visualizations. Model checkpoints remain outside Git.

## 7. Results

The first exact-SHA diagnostic pair at `6ea7f8e` completed before the
shared-initialization audit:

- official RWKV7: step-500 CE `1.0248`, mixed loop-5 exact `0.01367`,
  `46-50` exact `0.60742`, and exact `0` on `51-55` / `56-64`;
- official FLA GDN with matched `expand_v=1`: step-500 CE `1.0419`, mixed
  loop-5 exact `0.00977`, `46-50` exact `0.34766`, and exact `0` on
  `51-55` / `56-64`.

These two runs passed source, CUDA/Triton, data-order, gradient, runtime, and
artifact checks, but they are not accepted as the clean four-way baseline.
The independent initialization audit found that 11 of 77 same-shaped shared
parameter tensors differed by backbone: all ten ChannelMix input matrices and
the output head. Different backbone constructors consumed different numbers
of random draws before those shared modules were initialized. With one seed
and 500 steps, this is an avoidable nuisance variable.

At `2026-07-24T15:02:54Z`, the suite was restarted with an explicit
backbone-independent shared-shell initializer and a fail-closed GPU gate that
requires all shared parameter hashes to match. The `6ea7f8e` pair is retained
as diagnostic history and must not be used for the final carrier ranking.

Review note: before formal training, the checkpoint evaluator was corrected so
that `holes53` really selects exactly 53 blanks on official data. The previous
behavior ignored the requested count only for checkpoint evaluation; final
official blank-range evaluation was already correct. No result from the
mislabelled checkpoint path is used.

Review note: the aborted dirty-tree launch completed zero optimizer steps and
is not a benchmark arm. Its only admissible output is the infrastructure
failure record.

Review note: equal seed alone is not equal shared initialization when
architectures consume different random-number streams. Future one-seed carrier
comparisons must pass the shared-shell hash gate before training.

### Strict implementation audit

The accepted rerun passed all of the following before optimizer step 1:

- the 77 same-shaped shared-shell tensors were bit-identical across all four
  models;
- the official FLA wheel at commit
  `fe8fce9fc6984f22905f54cfa885dce1502baf26` matched the installed source
  files byte for byte;
- the instantiated classes were exactly FLA `GatedDeltaNet`,
  `GatedDeltaNet2`, and `KimiDeltaAttention`;
- the autograd graphs contained `ChunkGatedDeltaRuleFunctionBackward`,
  `ChunkGDN2FunctionBackward`, and `ChunkKDAFunctionBackward`;
- FLA output/reference errors were at most `6.6e-5`, terminal-state errors at
  most `2.8e-4`, and every initial recurrent state received a finite nonzero
  gradient;
- GDN and KDA used their official V-by-K cache layout; GDN2 used its official
  K-by-V layout;
- RWKV7 equations, initialization, `v_first`, CUDA state passing, and grouped
  AdamW matched the pinned official source;
- all arms consumed the same strict official-data stream and the same
  512-board evaluation sets.

After training, every arm separately passed
`strict_run_validation.json`. The final aggregator refuses dirty source,
nonempty patches, wrong classes/backends, missing validation, differing puzzle
hashes, or any metric that disagrees with the raw result.

### Accepted result

| Carrier | Params | CE@500 | Mixed exact L1->L5 | 46-50 exact L5 | 51-55 | 56-64 | Sec/step | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RWKV7 TimeMix | 5.093M | 1.0190 | 0.00% -> 1.95% | 72.85% | 0.00% | 0.00% | 4.04 | 10.78 |
| GDN, state-matched | 4.867M | 1.0509 | 0.20% -> 0.59% | 36.33% | 0.00% | 0.00% | 4.89 | 5.99 |
| GDN2 | 5.462M | 1.0062 | 1.37% -> 2.34% | 79.69% | 0.00% | 0.00% | 5.85 | 7.84 |
| KDA | 4.736M | 1.0179 | 0.20% -> 1.76% | 64.65% | 0.00% | 0.00% | 5.58 | 6.45 |

All four strict 53-blank evaluations are `0%` exact. On the same visualized
puzzle, wrong-cell counts across loops 1-5 are:

- RWKV7: `24,24,24,23,23`;
- GDN: `23,21,17,17,17`;
- GDN2: `21,19,20,20,20`;
- KDA: `28,23,24,24,24`.

The aggregate and all same-puzzle visualizations are archived under
`runs/sudoku-backbone-benchmark-sharedinit-20260724T154020Z-6e51f06/`.

## 8. Conclusions

1. There is no remaining evidence that GDN2 or KDA silently fell back or used
   the wrong recurrent-state orientation. Their official CUDA/Triton
   implementations pass reference, backward, cache-layout, source, and
   post-run gates.
2. The old claim that GDN clearly beats RWKV is withdrawn. Under the corrected
   shared initialization, official RWKV7 and GDN2 both open much better than
   the state-matched GDN arm.
3. GDN2 is the best finite-budget opener, not a hard-task winner. Every carrier
   is still `0%` exact at 51-64 blanks, so this experiment selects no universal
   architecture.
4. Most loop gain arrives by loop 2. Later loops mostly copy the same operating
   point, so this gate does not establish sustained recurrent correction.
5. This table deliberately matches recurrent-state size. It is not a
   native-optimal architecture comparison: official GDN recommends
   `num_heads * head_dim = 0.75 * hidden_size` with `expand_v=2`, while this
   state-matched arm uses K=V=32. One preregistered native-geometry GDN probe is
   justified to test whether configuration compression, rather than an
   implementation bug, caused the drop. It must remain separate from this
   table and must not become a geometry sweep.

## 9. Submission Record

No tag or submission is allowed unless the primary score is at least 0.50 and
the mechanism conclusion is clean.
