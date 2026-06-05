# 16x16 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 16x16, box: 4x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.1867, total=0.1867, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.5612, loop_last_loss=0.1867, sec=2663.1
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7713
  future_seed: fs_gate=0.507, fs_state_norm=8.107
- loop 2: exact=0.0312, valid=0.0312, solved=0.0312, blank_acc=0.9433
  future_seed: fs_gate=0.507, fs_state_norm=8.107
- loop 3: exact=0.0352, valid=0.0352, solved=0.0352, blank_acc=0.9437
  future_seed: fs_gate=0.507, fs_state_norm=8.107
- loop 4: exact=0.0430, valid=0.0430, solved=0.0430, blank_acc=0.9434
  future_seed: fs_gate=0.507, fs_state_norm=8.107
- loop 5: exact=0.0430, valid=0.0430, solved=0.0430, blank_acc=0.9432
  future_seed: fs_gate=0.507, fs_state_norm=8.107
- loop 6: exact=0.0430, valid=0.0430, solved=0.0430, blank_acc=0.9432
  future_seed: fs_gate=0.507, fs_state_norm=8.107
- loop 7: exact=0.0430, valid=0.0430, solved=0.0430, blank_acc=0.9432
  future_seed: fs_gate=0.507, fs_state_norm=8.107

## Stochastic Rollouts

- K1: oracle_exact=0.0430, oracle_solved=0.0430, disagree=0.0000
  confidence: exact=0.0430, valid=0.0430, solved=0.0430, blank_acc=0.9432; gap=0.0000
  consistency: exact=0.0430, valid=0.0430, solved=0.0430, blank_acc=0.9432; gap=0.0000
  residual: exact=0.0430, valid=0.0430, solved=0.0430, blank_acc=0.9432; gap=0.0000
  majority: exact=0.0430, valid=0.0430, solved=0.0430, blank_acc=0.9432; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0352, confidence=0.0352, residual=0.0352, disagree=0.0000
### loop5
- K1: oracle_exact=0.0430, confidence=0.0430, residual=0.0430, disagree=0.0000
### loop7
- K1: oracle_exact=0.0430, confidence=0.0430, residual=0.0430, disagree=0.0000

## Hole Transfer

- holes64: exact=0.0430, valid=0.0430, solved=0.0430, blank_acc=0.9432; indep=0.0237, exact/indep=1.82
- holes48: exact=0.4180, valid=0.4219, solved=0.4219, blank_acc=0.9766; indep=0.3216, exact/indep=1.30
- holes80: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.8724; indep=0.0000, exact/indep=0.00
- holes96: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7820; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/harder16-f170ff7/runs/frontier-16x16-unitstate-d256-l12-loop7-h80-b32-20260605T0658Z-f170ff7/output/futureseed_loop_case_seed52.html
