# 16x16 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 16x16, box: 4x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.1102, total=0.1102, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.2491, loop_last_loss=0.1102, sec=1407.9
- loop 1: exact=0.2031, valid=0.2031, solved=0.2031, blank_acc=0.9520
  future_seed: fs_gate=0.497, fs_state_norm=7.952
- loop 2: exact=0.6836, valid=0.6836, solved=0.6836, blank_acc=0.9866
  future_seed: fs_gate=0.497, fs_state_norm=7.952
- loop 3: exact=0.6758, valid=0.6758, solved=0.6758, blank_acc=0.9865
  future_seed: fs_gate=0.497, fs_state_norm=7.952
- loop 4: exact=0.6875, valid=0.6875, solved=0.6875, blank_acc=0.9871
  future_seed: fs_gate=0.497, fs_state_norm=7.952
- loop 5: exact=0.6914, valid=0.6914, solved=0.6914, blank_acc=0.9872
  future_seed: fs_gate=0.497, fs_state_norm=7.952

## Stochastic Rollouts

- K1: oracle_exact=0.6914, oracle_solved=0.6914, disagree=0.0000
  confidence: exact=0.6914, valid=0.6914, solved=0.6914, blank_acc=0.9872; gap=0.0000
  consistency: exact=0.6914, valid=0.6914, solved=0.6914, blank_acc=0.9872; gap=0.0000
  residual: exact=0.6914, valid=0.6914, solved=0.6914, blank_acc=0.9872; gap=0.0000
  majority: exact=0.6914, valid=0.6914, solved=0.6914, blank_acc=0.9872; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.2031, confidence=0.2031, residual=0.2031, disagree=0.0000
### loop3
- K1: oracle_exact=0.6758, confidence=0.6758, residual=0.6758, disagree=0.0000
### loop5
- K1: oracle_exact=0.6914, confidence=0.6914, residual=0.6914, disagree=0.0000

## Hole Transfer

- holes32: exact=0.6914, valid=0.6914, solved=0.6914, blank_acc=0.9872; indep=0.6618, exact/indep=1.04
- holes48: exact=0.2148, valid=0.2148, solved=0.2148, blank_acc=0.9631; indep=0.1648, exact/indep=1.30
- holes64: exact=0.0117, valid=0.0117, solved=0.0117, blank_acc=0.9156; indep=0.0035, exact/indep=3.31
- holes80: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.8386; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/unitstate-e9/runs/frontier-16x16-unitstate-d192-h64-20260605T0349Z-e9eabcf/output/futureseed_loop_case_seed52.html
