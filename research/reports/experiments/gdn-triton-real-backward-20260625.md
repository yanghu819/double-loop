# GDN Triton Real Backward Kernel

## 1. Metainfo

- run family: `gdn-realbwd-fs-s100-20260625T2130-645011a`
- plan ID: `P-GDN-004`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: `645011ae1fbecad71b6c6ce93a21d10d29e1b044`
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-25 UTC

## 2. Hypothesis

The previous local GDN Triton path was semantically correct, but backward still replayed the recurrence in PyTorch.

If FutureSeed is going to scale on Gated DeltaNet, that replay is the wrong bottleneck: it makes every larger gate test mostly a Python/autograd benchmark rather than a model experiment. A true Triton reverse-time backward should match the pure Torch recurrence and drop the small train gate time without changing model behavior.

This is infrastructure evidence, not a model-quality claim.

## 3. Configuration

- kernel: `experiments/rwkv_fs_sudoku/gdn_triton.py`
- recurrent state layout: native FutureSeed/GDN external layout `(B,H,V,K)`
- forward: saves per-step recurrent states only when gradients are needed
- backward: Triton reverse scan computes `dq`, `dk`, `dv`, `dg`, `dbeta`, and `dh0`
- reference checker: pure Torch recurrence in `check_gdn_triton_kernel.py --matrix`
- train gate: generated 9x9 Sudoku, random holes, D64/L2/H4/head_dim16, batch32, loops2, 100 steps
- FutureSeed: `FUTURE_SEED_SCALE=1.0`
- disabled mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep

This is a real Triton recurrent backward kernel. It is not yet a single fused whole-model training step.

## 4. Environment

- GPU row: GPU1
- GPU: NVIDIA A800-SXM4-80GB
- torch: `2.7.0+cu126`
- CUDA visible devices: `0`
- source SHA: `645011ae1fbecad71b6c6ce93a21d10d29e1b044`
- source dirty at run launch: `false`

## 5. Commands

Matrix checker:

```bash
PYTHONPATH=/huyang2/double-loop/.worktrees/gdn-real-bwd-clean-645011a-20260625T2130/experiments/rwkv_fs_sudoku \
/opt/conda/bin/python experiments/rwkv_fs_sudoku/check_gdn_triton_kernel.py \
  --matrix \
  --out /huyang2/double-loop/artifacts/gdn_real_bwd_clean_matrix_645011a.json
```

Train gate:

```bash
SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
PYTHON_BIN=/opt/conda/bin/python CUDA_VISIBLE_DEVICES=0 \
SUDOKU_SIZE=9 HOLE_PATTERN=random HOLE_STAGES=8-16:50,16-24:50 \
EVAL_HOLES=16 EVAL_HOLES_LIST=8,16,24 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=1.0 \
D_MODEL=64 LAYERS=2 HEADS=4 HEAD_DIM=16 CHANNEL_MULT=2 L_CYCLES=1 MAX_LOOPS=2 \
FULL_STEPS=100 FULL_BATCH=32 FULL_EVAL_N=256 FULL_ROLLOUT_KS= FULL_LOG_EVERY=25 \
FORWARD_DTYPE=bfloat16 LR=0.002 BLANK_LOSS_WEIGHT=20 FUTURE_SEED_SCALE=1.0 \
RUN_NAME=gdn-realbwd-fs-s100-20260625T2130-645011a ./run.sh full
```

Launch script:

```bash
/huyang2/double-loop/artifacts/launch/gdn_real_bwd_clean_645011a_20260625T2130.sh
```

## 6. Artifacts

- matrix: `artifacts/gdn_real_bwd_clean_matrix_645011a.json`
- run: `runs/gdn-realbwd-fs-s100-20260625T2130-645011a`
- launch: `artifacts/launch/gdn_real_bwd_clean_645011a_20260625T2130.sh`
- log: `artifacts/logs/gdn_real_bwd_clean_645011a_20260625T2130.log`
- pid file: `artifacts/logs/gdn_real_bwd_clean_645011a_20260625T2130.pid`

The run directory includes `config.json`, `score.json`, `logs/run.log`, output JSON/MD/HTML, `source_HEAD.txt`, `source.patch`, and `source_snapshot.tar.gz`.

## 7. Results

Kernel alignment matrix:

| item | value |
|---|---:|
| cases passed | `7 / 7` |
| failed cases | `0` |
| max output abs diff | `5.9605e-08` |
| max final-state abs diff | `3.5763e-07` |
| max grad q abs diff | `5.0664e-07` |
| max grad k abs diff | `2.3842e-06` |
| max grad v abs diff | `2.3842e-07` |
| max grad g abs diff | `3.8147e-06` |
| max grad beta abs diff | `2.8610e-06` |
| max grad h0 abs diff | `5.9605e-07` |
| min forward speedup in matrix | `10.83x` |

Clean FutureSeed train gate:

| metric | value |
|---|---:|
| train seconds | `10.05` |
| previous replay-backward FS gate seconds | `69.10` |
| speedup vs replay gate | `6.87x` |
| train CE | `1.6118` |
| loop2 exact | `0.0000` |
| loop2 blank_acc | `0.2385` |
| h8/h16/h24 loop2 blank_acc | `0.2666 / 0.2385 / 0.2023` |
| fs_gate | `0.4713` |
| fs_state_norm | `7.5400` |
| CUDA max allocated MB | `221.19` |

## 8. Conclusions

The real backward kernel works.

What this proves:

- The GDN recurrent training path no longer depends on PyTorch replay inside custom autograd backward.
- Gradients match the pure Torch recurrence across the existing CUDA matrix, including initial-state, state-loss, non-power dimensions, V64, longer T32/K32, and bf16 runner-like cases.
- The small FS train gate drops from `69.10s` with replay backward to `10.05s` with the real Triton backward, while preserving the same tiny-run behavior.
- Native FutureSeed state seeding remains active: `fs_gate=0.4713`, `fs_state_norm=7.5400`.

What this does not prove:

- It does not prove GDN+FutureSeed model quality. The 100-step D64/L2 gate is intentionally too small and exact remains `0`.
- It does not prove the final fastest architecture. The short convolution is still off, and this is not a whole-model fused optimizer step.

Decision:

- Keep the real Triton backward path.
- Stop tiny kernel gates unless a future code change touches the recurrence.
- Next high-ROI experiment is a larger matched GDN no-FS vs native-FS scale gate, or an official Sudoku GDN gate if the data path is ready.
- If memory becomes the next bottleneck, optimize state saving or add checkpoint/recompute for the saved recurrent states.

## 9. Submission Record

None.
