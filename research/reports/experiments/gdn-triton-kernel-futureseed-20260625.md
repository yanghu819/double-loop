# GDN Triton FutureSeed Recurrent Kernel

## 1. Metainfo

- run family: `gdn-triton-kernel-check-20260625`
- plan ID: `P-GDN-002`
- machine: AIStation GPU1 only
- remote work dir: `/huyang2/double-loop`
- local branch: `codex/gpu1-experiment-tracking`
- implementation time: 2026-06-25 Asia/Shanghai

## 2. Hypothesis

The previous GDN gate was blocked by engineering, not by the FutureSeed idea itself: FLA chunk forward worked, but chunk backward stalled, while FLA naive recurrent trained but was too slow to scale.

A minimal FutureSeed-aware GDN recurrent state kernel should directly operate on our external state layout `(B,H,V,K)`, so terminal state from one layer can seed the next layer without layout hacks. The first bar is correctness and trainability, not benchmark quality.

## 3. Configuration

- new kernel file: `experiments/rwkv_fs_sudoku/gdn_triton.py`
- runner mode: `BACKBONE=gdn GDN_MODE=triton_recurrent`
- state layout: external FutureSeed/GDN state is `(B,H,V,K)`
- recurrence: Gated DeltaNet update with normalized q/k, log decay `g`, sigmoid beta, optional initial state, and terminal state output.
- backward: custom autograd function with Triton forward and PyTorch replay backward. This keeps gradients correct while leaving the high-performance backward kernel for the next step.
- forbidden mechanisms: no reverse scan, no selector, no repair, no Sudoku rule, no CPU fallback.

## 4. Environment

- GPU row: GPU1
- GPU: NVIDIA A800-SXM4-80GB
- torch: `2.7.0+cu126`
- execution: CUDA only

## 5. Commands

Micro correctness check:

```bash
PYTHONPATH=/huyang2/double-loop/.worktrees/gdn-fla-9336e94-20260625T1708/experiments/rwkv_fs_sudoku \
/opt/conda/bin/python experiments/rwkv_fs_sudoku/check_gdn_triton_kernel.py \
  --batch 2 --seq 9 --heads 3 --dim 16 --value_dim 16 \
  --out /huyang2/double-loop/artifacts/gdn_triton_microcheck.json
```

Matrix correctness check:

```bash
PYTHONPATH=/huyang2/double-loop/.worktrees/gdn-fla-9336e94-20260625T1708/experiments/rwkv_fs_sudoku \
/opt/conda/bin/python experiments/rwkv_fs_sudoku/check_gdn_triton_kernel.py \
  --matrix \
  --out /huyang2/double-loop/artifacts/gdn_triton_matrix_check.json
```

Direct runner check:

```bash
bash /huyang2/double-loop/artifacts/launch/start_gdn_triton_direct_tiny_bg.sh
```

The direct runner intentionally bypassed `run.sh` because the remote worktree repeatedly hit D-state in git/source snapshot setup after AIStation restart. It still used GPU1/CUDA and the same Python runner.

## 6. Artifacts

- microcheck: `runs/gdn-triton-kernel-check-20260625/microcheck.json`
- matrix check: `runs/gdn-triton-kernel-check-20260625/matrix_check.json`
- direct runner log: `runs/gdn-triton-kernel-check-20260625/direct_tiny.log`
- direct runner output: `runs/gdn-triton-kernel-check-20260625/direct_tiny`
- launch scripts: `artifacts/launch/run_gdn_triton_direct_tiny.sh`, `artifacts/launch/start_gdn_triton_direct_tiny_bg.sh`

## 7. Results

Microcheck, Triton forward versus PyTorch replay:

| metric | value |
|---|---:|
| forward output max diff | `4.470348e-08` |
| final state max diff | `2.980232e-07` |
| grad q/k/v/g/beta/h0 max diff | `0.0` |
| Triton forward sec | `0.000432` |
| PyTorch forward sec | `0.057367` |

Matrix check, Triton forward versus PyTorch replay:

| coverage | result |
|---|---:|
| cases passed | `7 / 7` |
| max output abs diff | `5.960464e-08` |
| max final-state abs diff | `3.576279e-07` |
| max grad abs diff for q/k/v/g/beta/h0 | `0.0` |
| max output relative diff | `6.369427e-04` |
| max final-state relative diff | `2.339321e-03` |
| minimum forward speedup vs PyTorch loop | `8.7397x` |

The matrix cases cover:

- no initial state and `T=1`
- normal output-plus-state loss with `(B,T,H,K,V)=(2,9,3,16,16)`
- non-power dimensions `(K,V)=(12,20)` with strong decay
- state-only loss with no initial state
- longer `T=32`, `K=32`
- `V=64` boundary
- bf16 runner-like inputs

Two test-harness lessons were fixed during this pass:

- State-only losses can leave some PyTorch reference gradients unused; the checker now treats unused gradients as zeros on both sides.
- Near-zero tensors can have noisy relative error even when absolute error is tiny; the pass rule now fails only when both absolute and relative thresholds are exceeded.

Direct runner CUDA tiny train:

| step | CE |
|---:|---:|
| 1 | `2.4810` |
| 2 | `2.4179` |

Final tiny eval:

- loop1 exact `0.0`, blank_acc `0.1250`
- loop2 exact `0.0`, blank_acc `0.1250`
- fs_gate `0.500`, fs_state_norm `8.000`
- train_sec `3.67`
- peak allocated memory `26 MB`

This is an implementation smoke result only. It is intentionally too small to judge model quality.

## 8. Conclusions

Keep the implementation.

What is solid:

- The local Triton recurrent kernel matches the PyTorch reference in forward output, terminal state, and gradients across a small but targeted CUDA matrix.
- The kernel preserves native FutureSeed semantics: terminal recurrent state in `(B,H,V,K)` seeds the next layer. No right-to-left scan or task-specific rule was added.
- The full runner can train and evaluate with `GDN_MODE=triton_recurrent` on GPU1.

What is still not solved:

- Backward is currently PyTorch replay, not a Triton backward kernel. It is correct but not yet the final scaling path.
- `run.sh` can still hang in remote source snapshot on this worktree after AIStation restart; direct Python runner was used for this check.
- This does not prove GDN+FutureSeed model quality. It only removes the first kernel/trainability blocker.

Next:

- Write the real Triton backward or a fused recurrent training kernel for the `(B,H,V,K)` state layout, using the matrix checker as the regression gate.
- Then rerun a matched generated 9x9 no-FS vs FS gate at D64/L2 or D96/L4 and compare speed to FLA naive.
- Only after that, move back to official Sudoku scale.

## 9. Submission Record

None.
