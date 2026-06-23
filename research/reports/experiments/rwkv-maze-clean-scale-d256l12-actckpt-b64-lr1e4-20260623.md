# rwkv-maze-clean-scale-d256l12-actckpt-b64-lr1e4-20260623

## 1. Metainfo

- Plan ID: P-MAZE-008
- Status: discarded
- Machine: AIStation GPU1 only
- Start timestamp: 2026-06-23T00:32:20Z
- Local branch: `codex/gpu1-experiment-tracking`
- Source SHA: `68845e4979efebcb30232a52297239367ebf84df`

## 2. Hypothesis

The batch64 activation-checkpoint run fit and used GPU, but `lr=3e-4` produced
NaNs by step100. That is an optimizer stability failure, not yet a mechanism
answer. A single lower-LR scale probe tests whether pure compute can remain
stable long enough to show real loop pruning.

This is not a learning-rate sweep. It is the minimum stabilization needed to
make the chosen scale point trainable.

## 3. Configuration

- Dataset: official `maze-30x30-unique-1k`
- Runner: `scripts/rwkv_maze_probe.py`
- Objective: clean all-loop CE with PATH weight `8`
- FutureSeed: enabled, `future_seed_scale=1`
- Feedback/DAT: disabled
- Boundary/budget decoder: disabled
- Model: D256/L12/H8/head_dim32/channel_mult4
- Loops: train loops 8, eval loops 16
- Batch/eval: batch64, eval_n256, eval_batch16
- Activation checkpointing: enabled
- LR: `1e-4`
- Steps: 800, log every 100
- Early gate: at step800, abort if loop16 precision remains below `0.38` and
  loop16 has not dropped at least `10` FP/case relative to loop1.

## 4. Environment

- Remote base: `/huyang2/double-loop`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- Python: `/opt/conda/bin/python`
- GPU: AIStation GPU1, A800 80GB
- GPU2: forbidden
- CPU smoke: forbidden
- Cache/artifacts/runs: project-local under `/huyang2/double-loop`

## 5. Commands

Launched from:

```bash
bash /huyang2/double-loop/artifacts/launch/run_rwkv_maze_clean_scale_d256l12_actckpt_b64_lr1e4_20260623.sh
```

The script archives the GitHub-truth SHA into
`/huyang2/double-loop/artifacts/source/rwkv-maze-clean-scale-d256l12-20260623-68845e4`
and writes logs under
`/huyang2/double-loop/runs/rwkv-maze-clean-scale-d256l12-actckpt-b64-lr1e4-20260623`.

## 6. Artifacts

Actual:

- `runs/rwkv-maze-clean-scale-d256l12-actckpt-b64-lr1e4-20260623/abort.json`
- `runs/rwkv-maze-clean-scale-d256l12-actckpt-b64-lr1e4-20260623/launch.log`
- `runs/rwkv-maze-clean-scale-d256l12-actckpt-b64-lr1e4-20260623/launch.env`
- `runs/maze-scale-summary-20260623/index.html`

The run was manually stopped at step200 for a low-ROI broad-mask plateau, so it
does not have model-generated hard-case HTML.

## 7. Results

| step | CE | loop1 F1 | loop16 F1 | gain | pred frac loop1->16 | FP loop1->16 | FN loop1->16 |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 1.8253 | 0.3124 | 0.3008 | -0.0115 | 0.2603->0.2333 | 178.9->160.3 | 63.9->69.6 |
| 100 | 0.8921 | 0.2968 | 0.3140 | +0.0172 | 0.7596->0.7106 | 564.3->520.2 | 0.0->0.0 |
| 200 | 0.9004 | 0.2972 | 0.3144 | +0.0171 | 0.7583->0.7097 | 563.2->519.4 | 0.0->0.0 |

Important readout:

- Lower LR fixed the NaN issue from `lr=3e-4`.
- Loop16 consistently reduced false positives by about `44` cells/case at
  step100-200 while preserving zero false negatives.
- The prediction mask stayed very broad: loop16 pred path frac stayed near
  `0.71`.
- Step100 to step200 showed no meaningful improvement, so continuing to step800
  was judged low ROI.

## 8. Conclusions

Pure scale is not a complete Maze fix in the current objective/model setup.
This run gives a useful positive mechanism sliver: loop can prune false
positives without creating false negatives when the large-batch run is stable.
But the model remains locked into a high-recall broad-mask attractor, and more
steps at the same operating point were unlikely to answer a new question.

Decision: stop this pure-scale branch as a primary Maze path. The next high-ROI
work should keep the bitter-lesson constraints but change the generic training
signal or decision state so the existing loop-pruning signal can move the hard
mask boundary much farther.

## 9. Submission Record

None.
