# rwkv-maze-abortviz-probe-20260622

## 1. Metainfo

- Plan ID: P-MAZE-007
- Status: in progress
- Machine: AIStation GPU1 only
- Start timestamp: 2026-06-22T14:29:06Z
- Local branch: `codex/gpu1-experiment-tracking`
- Parent SHA before instrumentation: `f1c11e67b49ce412efbeb4d5df4108dcd55e90bf`

## 2. Hypothesis

The latest Maze DAT result is already negative, but its kill path did not preserve
hard-case loop visuals. Before spending more GPU on new state dynamics, abortable
Maze probes must leave enough evidence to see whether loops are adding false
positives, pruning true path cells, or copying the same broad mask.

This is an instrumentation probe, not a new positive mechanism claim.

## 3. Configuration

- Dataset: official `maze-30x30-unique-1k`
- Runner: `scripts/rwkv_maze_probe.py`
- Model: causal RWKV/FutureSeed Maze runner, D128/L8
- FutureSeed: enabled, `future_seed_scale=1`
- Feedback/DAT: same generic DAT path as the previous negative run
- Eval loops: 16
- Visualization: input, target, loop1, loop4, loop8, loop16
- Broad-mask abort gate: enabled with a strict precision threshold to verify the
  archive path quickly

## 4. Environment

- Remote work dir: `/huyang2/double-loop`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- CPU smoke: forbidden
- GPU2: forbidden
- Cache/artifacts/runs: project-local under `/huyang2/double-loop`

## 5. Commands

Exact launch command will be recorded from the generated launch script after the
GitHub-truth SHA is committed and checked out on GPU1.

## 6. Artifacts

Expected:

- `runs/rwkv-maze-abortviz-probe-20260622/rwkv_maze_probe.json`
- `runs/rwkv-maze-abortviz-probe-20260622/abort.json`
- `runs/rwkv-maze-abortviz-probe-20260622/README.md`
- `runs/rwkv-maze-abortviz-probe-20260622/visualizations/index.html`
- `runs/rwkv-maze-abortviz-probe-20260622/visualizations/cases.json`

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

None. This is not a submission or score-bearing paper result.
