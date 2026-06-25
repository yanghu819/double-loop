# GDN Triton Recurrent Train Gate

## 1. Metainfo

- run family: `gdn-triton-*-s100-20260625T2035-0f5771c`
- plan ID: `P-GDN-003`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: `0f5771c8557ca757a3e7fd1bd2b16e5f70bf44fc`
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-25 UTC

## 2. Hypothesis

The previous kernel work proved local GDN Triton recurrent semantics, but not yet an end-to-end training path.

The question here is narrow: can `GDN_MODE=triton_recurrent` train, evaluate, record, and archive a matched no-FutureSeed/FutureSeed pair from a clean Git SHA on GPU1?

This is not a model-quality experiment. It is a gate before investing in a real Triton backward/fused train kernel.

## 3. Configuration

- task: generated 9x9 Sudoku, random holes
- stages: `8-16:50,16-24:50`
- eval holes: `8,16,24`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0` to avoid routing through FLA for this kernel gate
- size: `D_MODEL=64`, `LAYERS=2`, `HEADS=4`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=2`
- batch: `32`
- steps: `100`
- dtype: `bfloat16`
- arms:
  - no-FS: `FUTURE_SEED_SCALE=0`
  - FS: `FUTURE_SEED_SCALE=1`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep.

## 4. Environment

- GPU row: GPU1
- GPU: NVIDIA A800-SXM4-80GB
- torch: `2.7.0+cu126`
- Triton: `3.3.0`
- CUDA visible devices: `0`

## 5. Commands

Launch script:

```bash
/huyang2/double-loop/artifacts/launch/gdn_triton_gate_20260625T2035.sh
```

The script created two clean detached worktrees at `0f5771c` and ran:

```bash
SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
PYTHON_BIN=/opt/conda/bin/python CUDA_VISIBLE_DEVICES=0 \
SUDOKU_SIZE=9 HOLE_PATTERN=random HOLE_STAGES=8-16:50,16-24:50 \
EVAL_HOLES=16 EVAL_HOLES_LIST=8,16,24 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=1.0 \
D_MODEL=64 LAYERS=2 HEADS=4 HEAD_DIM=16 CHANNEL_MULT=2 L_CYCLES=1 MAX_LOOPS=2 \
FULL_STEPS=100 FULL_BATCH=32 FULL_EVAL_N=256 FULL_ROLLOUT_KS= FULL_LOG_EVERY=25 \
FORWARD_DTYPE=bfloat16 LR=0.002 BLANK_LOSS_WEIGHT=20 \
FUTURE_SEED_SCALE=<0-or-1> ./run.sh full
```

## 6. Artifacts

- no-FS run: `runs/gdn-triton-nofs-s100-20260625T2035-0f5771c`
- FS run: `runs/gdn-triton-fs-s100-20260625T2035-0f5771c`
- launch script: `artifacts/launch/gdn_triton_gate_20260625T2035.sh`
- combined remote log: `artifacts/logs/gdn_triton_gate_20260625T2035.log`

Both run directories include `config.json`, `score.json`, `logs/run.log`, output JSON/MD/HTML, `source_HEAD.txt`, `source.patch`, and `source_snapshot.tar.gz`.

## 7. Results

Training:

| arm | train sec | train CE | loop1 loss | loop2 loss | CUDA alloc MB |
|---|---:|---:|---:|---:|---:|
| no-FS | `75.77` | `1.6005` | `1.5980` | `1.6005` | `150.0` |
| FS | `69.10` | `1.6116` | `1.6077` | `1.6116` | `151.5` |

Clean eval:

| arm | loop1 exact | loop1 blank_acc | loop2 exact | loop2 blank_acc | fs_gate | fs_state_norm |
|---|---:|---:|---:|---:|---:|---:|
| no-FS | `0.0000` | `0.2354` | `0.0000` | `0.2344` | `0.000` | `0.000` |
| FS | `0.0000` | `0.2417` | `0.0000` | `0.2385` | `0.471` | `7.541` |

Hole transfer, loop2 blank accuracy:

| arm | h8 | h16 | h24 |
|---|---:|---:|---:|
| no-FS | `0.2700` | `0.2344` | `0.1950` |
| FS | `0.2681` | `0.2385` | `0.2010` |

## 8. Conclusions

The path is now genuinely runnable.

What this proves:

- `GDN_MODE=triton_recurrent` can run complete CUDA train/eval/record from clean SHA.
- Native FutureSeed terminal-state seeding is active in GDN: `fs_gate≈0.471`, `fs_state_norm≈7.54`.
- The run archives normal provenance and visual artifacts, not just stdout.

What this does not prove:

- It does not prove GDN+FutureSeed quality. The gate is only 100 steps at D64/L2 and both arms have exact `0`.
- It does not prove final scaling speed. Backward still replays the PyTorch recurrence, so GPU utilization and throughput are not the final story.
- The short convolution is off, so this gate isolates our recurrent kernel path rather than matching FLA GDN architecture exactly.

Decision:

- Keep the GDN Triton path.
- Stop tiny end-to-end gates; they now have low information gain.
- Next high-ROI work is a real Triton backward/fused train kernel using the existing matrix checker as the regression gate.
- After backward is real, run a larger matched no-FS vs FS gate before making any model-quality claim.

## 9. Submission Record

None.
