# FutureSeed Causal Four-Backbone Comparison

## 1. Metainfo

- Experiment: `E-BASELINE-005`
- Plan: `P-BASELINE-005`
- Status: in progress
- Scheduled: 2026-07-26 10:27 CST / 2026-07-26 02:27 UTC
- Machine: AIStation GPU1, NVIDIA A800-SXM4-80GB
- GPU2: forbidden and halted
- CPU model smoke: forbidden
- Formal source: `6e51f067a18d936bda7f3d588b7b3f32625b2b18`
- Suite: `nofs-causal-20260726T022718Z-6e51f06`

## 2. Hypothesis

The accepted strict table ranks four FutureSeed-enabled recurrent carriers but
cannot estimate FutureSeed's causal contribution.

If FutureSeed is a generic cheap-future-context mechanism, enabling it should
improve full-board exactness across at least two independently implemented
causal recurrent carriers under identical initialization, data, objective,
optimizer, recurrent-state geometry, loop budget, and evaluator. A blank
accuracy increase without full-board exact improvement does not pass.

This is the minimum causal comparison required by the paper claim. It is not a
seed, geometry, loss, LR, or budget sweep.

## 3. Configuration

Both FutureSeed states use:

- carriers: official RWKV7 TimeMix, official FLA GDN, GDN2, and KDA;
- D192, 10 layers, 6 heads, head dimension 32;
- matched six-head 32-by-32 recurrent state;
- five loops and equal cross-entropy supervision on every loop;
- microbatch 32, accumulation 4, effective batch 128;
- BF16, AdamW, LR 0.0015, weight decay 0.001;
- seed 52 and backbone-independent shared-shell initialization seed 52;
- official full-diversity Sudoku and curriculum `46-50:100,51-55:400`;
- identical fixed-53, mixed, and official 46-50/51-55/56-64 evaluation;
- no noise, feedback, scratch state, selector, repair, search, or task rule.

The sole substantive intervention is `FUTURE_SEED_SCALE=1` versus `0`.
Run/checkpoint output paths necessarily differ. Parameter counts must match
within each carrier.

## 4. Environment

- Persistent root: `/huyang2/double-loop`
- Formal detached worktree:
  `/huyang2/double-loop/artifacts/worktrees/fs-causal-6e51f06`
- Python: `/opt/conda/bin/python`
- Python extras: `/huyang2/double-loop/.cache/python-extra-pylib`
- Data: `/huyang2/double-loop/data/sudoku-extreme-full`
- Models: `/huyang2/double-loop/models`
- Runs: `/huyang2/double-loop/runs`
- Required device: `CUDA_VISIBLE_DEVICES=0`
- FLA: backend dispatch disabled, Triton convolution required

## 5. Commands

Preflight and formal launch are archived under:

`/huyang2/double-loop/artifacts/launch/nofs-causal-20260726T022718Z-6e51f06/`

Each no-FutureSeed arm runs:

```bash
CUDA_VISIBLE_DEVICES=0 \
BENCHMARK_CONFIG=<archived-nofs-config> \
BENCHMARK_STEPS=500 \
PERSIST_ROOT=/huyang2/double-loop \
RUNS_ROOT=/huyang2/double-loop/runs \
PYTHON_BIN=/opt/conda/bin/python \
./scripts/run_sudoku_backbone_benchmark.sh <rwkv|gdn|gdn2|kda> <run-name>
```

Kill conditions:

- any configuration difference beyond FutureSeed scale and output paths;
- source, formula, CUDA/Triton backward, state, shared initialization, data,
  evaluator, parameter-count, or no-fallback gate fails;
- step 100 takes more than 15 minutes;
- NaN, OOM, wrong GPU, or low utilization with high memory.

There is no automatic kernel, batch, device, or implementation fallback.

## 6. Artifacts

Pending.

## 7. Results

Pending.

Success requires at least two carriers with official 46-50 blank loop-5 exact
gain `FS - noFS >= 0.10`, plus loop/case evidence that the gain is not only an
isolated operating-point fluctuation. A stronger result is FutureSeed reaching
the no-FutureSeed loop-5 quality in fewer loops.

## 8. Conclusions

Pending.

## 9. Submission Record

No tag or submission is allowed unless the primary score is at least 0.50 and
the mechanism conclusion is clean.
