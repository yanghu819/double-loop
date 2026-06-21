# rwkv-maze-official-fs-vs-nofs-20260621

## 1. Metainfo

- Plan ID: P-MAZE-003
- Status: discarded
- Machine: GPU1 A100
- Start time UTC: 2026-06-21T12:28:00Z
- Source commit: 1ad39c4
- Remote work dir: /huyang2/double-loop/.worktrees/rwkv-maze-archive-1ad39c4

## 2. Hypothesis

If FutureSeed is useful because it cheaply supplies future context to a causal
recurrent backbone, then a causal RWKV7 Maze model should show a with/without
FutureSeed gap on official path recovery.

## 3. Configuration

- Data: official `maze-30x30-unique-1k`
- Model: causal RWKV7 state-passing token classifier
- D_MODEL: 128
- Layers: 8
- Heads/head dim: 8 / 16
- Train/eval loops: 4 / 8
- Batch: 32
- Steps: 800
- Objective: all-loop token CE with PATH token weight `8`
- Conditions: `FUTURE_SEED_SCALE=0` and `FUTURE_SEED_SCALE=1`
- Forbidden: selector, search, repair, maze rules, postprocessing

## 4. Environment

- AIStation row: GPU1 only
- Python: /opt/conda/bin/python
- CUDA binding: `CUDA_VISIBLE_DEVICES=0`
- RWKV extension cache: /huyang2/double-loop/.cache/torch_extensions

## 5. Commands

Both runs were launched directly from the archived Git source tree at SHA
`1ad39c4`; see run logs in the local archive for full commands.

## 6. Artifacts

- Local archive: runs/rwkv-maze-official-fs-vs-nofs-20260621
- No-FutureSeed HTML: runs/rwkv-maze-official-fs-vs-nofs-20260621/nofs/visualizations/index.html
- FutureSeed HTML: runs/rwkv-maze-official-fs-vs-nofs-20260621/futureseed/visualizations/index.html
- Comparison JSON: runs/rwkv-maze-official-fs-vs-nofs-20260621/comparison.json
- Score JSON: runs/rwkv-maze-official-fs-vs-nofs-20260621/score.json

## 7. Results

| condition | loop8 path F1 | loop gain | precision | recall | pred PATH frac | FP/case | FN/case |
|---|---:|---:|---:|---:|---:|---:|---:|
| RWKV no FutureSeed | 0.4690 | +0.0000 | 0.3119 | 0.9580 | 0.4065 | 252.0 | 5.0 |
| RWKV + FutureSeed | 0.4666 | -0.0007 | 0.3111 | 0.9464 | 0.4026 | 249.8 | 6.4 |

## 8. Conclusions

Discard this as a positive FutureSeed claim. The causal RWKV backbone reaches
the same broad-mask solution as official EqR under the path-weighted objective.
FutureSeed does not improve it, and loops do not perform meaningful correction.

This is useful because it identifies the benchmark/objective as the bottleneck:
official Maze path-weight rewards high-recall coverage and does not force the
model to choose the unique path. The next experiment should make broad false
positives costly through a generic objective or choose a proxy where broad
coverage is not a cheap solution.

## 9. Submission Record

Not applicable.
