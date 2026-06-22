# rwkv-maze-abortviz-probe-20260622

## 1. Metainfo

- Plan ID: P-MAZE-007
- Status: done
- Machine: AIStation GPU1 only
- Start timestamp: 2026-06-22T14:29:06Z
- Local branch: `codex/gpu1-experiment-tracking`
- Parent SHA before instrumentation: `f1c11e67b49ce412efbeb4d5df4108dcd55e90bf`
- Instrumentation SHA: `328a08cdcaaf27b10a5653122578c10d8fe7d766`

## 2. Hypothesis

The latest Maze DAT result is already negative, but its kill path did not preserve
hard-case loop visuals. Before spending more GPU on new state dynamics, abortable
Maze probes must leave enough evidence to see whether loops are adding false
positives, pruning true path cells, or copying the same broad mask.

This is an instrumentation probe, not a new positive mechanism claim.

## 3. Configuration

- Dataset: official `maze-30x30-unique-1k`
- Data path: `/huyang2/double-loop/official_eqr_compare/eqr-clean/data/maze-30x30-unique-1k`
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

GitHub-truth code was committed and pushed as
`328a08cdcaaf27b10a5653122578c10d8fe7d766`.

Primary launch attempt used a detached worktree:

```bash
bash /huyang2/double-loop/artifacts/launch/run_rwkv_maze_abortviz_20260622.sh
```

That path stalled inside `git worktree add` / `git reset --hard` because the
remote repository has many historical tracked artifacts. The exact setup PIDs
were stopped and `git worktree prune` was run.

Fallback launch used a minimal Git archive from the same remote object, limited
to the runner source and RWKV implementation:

```bash
nohup bash /huyang2/double-loop/artifacts/launch/run_rwkv_maze_abortviz_archive_20260622.sh \
  > /huyang2/double-loop/runs/rwkv-maze-abortviz-probe-20260622/setup.log 2>&1 &
```

The actual Python command inside that script was:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python scripts/rwkv_maze_probe.py \
  --repo-root /huyang2/double-loop/artifacts/source/rwkv-maze-abortviz-minimal-20260622T1448Z-328a08c \
  --data-dir /huyang2/double-loop/official_eqr_compare/eqr-clean/data/maze-30x30-unique-1k \
  --out-dir /huyang2/double-loop/runs/rwkv-maze-abortviz-probe-20260622 \
  --run-name rwkv-maze-abortviz-probe-20260622 \
  --condition fs-dat-abortviz \
  --steps 300 --batch 16 --eval-n 128 --eval-batch 32 \
  --d-model 128 --layers 8 --heads 8 --head-dim 16 --channel-mult 4 \
  --l-cycles 2 --train-loops 4 --eval-loops 16 \
  --future-seed-scale 1 --feedback-mode pred \
  --rwkv-kernel statepassing --forward-dtype bfloat16 \
  --path-weight 8 --dat-weight 0.5 --dat-loops 2 --loop-loss all \
  --seed 52 --log-every 100 --viz-cases 32 \
  --broad-mask-kill-step 100 --broad-mask-min-precision 0.55 \
  --broad-mask-min-fp-drop 0 --broad-mask-target raw
```

## 6. Artifacts

- `runs/rwkv-maze-abortviz-probe-20260622/rwkv_maze_probe.json`
- `runs/rwkv-maze-abortviz-probe-20260622/abort.json`
- `runs/rwkv-maze-abortviz-probe-20260622/README.md`
- `runs/rwkv-maze-abortviz-probe-20260622/visualizations/index.html`
- `runs/rwkv-maze-abortviz-probe-20260622/visualizations/cases.json`
- `runs/rwkv-maze-abortviz-probe-20260622/launch.env`
- `runs/rwkv-maze-abortviz-probe-20260622/launch.log`
- `runs/rwkv-maze-abortviz-probe-20260622/source_sha.txt`

## 7. Results

The probe aborted at step100 by the intended broad-mask gate:

| step | loop1 F1 | loop16 F1 | loop gain | loop1 precision | loop16 precision | loop1 recall | loop16 recall | loop1 pred frac | loop16 pred frac | loop1 FP | loop16 FP | loop1 FN | loop16 FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3081 | 0.3111 | +0.0031 | 0.1906 | 0.1936 | 0.8087 | 0.7981 | 0.5513 | 0.5358 | 401.6 | 388.9 | 22.3 | 23.6 |
| 100 | 0.4614 | 0.4614 | -0.0000 | 0.3006 | 0.3006 | 1.0000 | 1.0000 | 0.4322 | 0.4322 | 272.1 | 272.1 | 0.0 | 0.0 |

Abort reason:

```json
{
  "reason": "broad_mask_low_precision_no_fp_drop",
  "step": 100,
  "precision": 0.3005557358264923,
  "loop1_fp": 272.078125,
  "loop_last_fp": 272.078125,
  "fp_drop": 0.0
}
```

Hard-case visual selection succeeded. The eight selected cases all show loop16
copying the broad mask from loop1:

| case | loop1 F1 | loop16 F1 | loop1 FP | loop16 FP | loop1 FN | loop16 FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.406 | 0.406 | 290 | 290 | 0 | 0 |
| 16 | 0.412 | 0.412 | 288 | 288 | 0 | 0 |
| 10 | 0.415 | 0.415 | 287 | 287 | 0 | 0 |
| 6 | 0.422 | 0.422 | 285 | 285 | 0 | 0 |
| 27 | 0.425 | 0.425 | 284 | 284 | 0 | 0 |
| 1 | 0.431 | 0.431 | 282 | 282 | 0 | 0 |
| 18 | 0.431 | 0.431 | 282 | 282 | 0 | 0 |
| 13 | 0.435 | 0.435 | 281 | 281 | 0 | 0 |

## 8. Conclusions

Keep the instrumentation. It solved the concrete evidence gap: aborted Maze
probes now produce `abort.json`, partial metrics, and loop hard-case visuals.

Do not treat this run as model progress. The model result is another clean
negative for the current DAT path: by step100, loop16 is exactly the same
broad-mask operating point as loop1. This strengthens the existing conclusion
that official Maze path-weight recovery is dominated by high-recall broad masks,
and that future Maze work must change the recurrent decision/state variable or
training pressure more substantially.

Operational lesson: remote full worktree/source snapshot can be too heavy because
the repository contains many historical tracked artifacts. For tiny probes, a
minimal Git archive of only the required source paths is a cleaner fallback while
still binding the run to the pushed SHA.

## 9. Submission Record

None. This is not a submission or score-bearing paper result.
