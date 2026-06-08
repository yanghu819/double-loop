# 16x16 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 16x16, box: 4x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9825, total=0.9829, loop_loss=all, noise=feature_diff, buffer=8192, loop1_loss=0.9865, loop_last_loss=0.9825, sec=851.7
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3312
  future_seed: fs_gate=0.499, fs_state_norm=15.956, fs_decay=0.00
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3359
  future_seed: fs_gate=0.499, fs_state_norm=15.956, fs_decay=0.00
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3376
  future_seed: fs_gate=0.499, fs_state_norm=15.956, fs_decay=0.00
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3377
  future_seed: fs_gate=0.499, fs_state_norm=15.956, fs_decay=0.00
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3377
  future_seed: fs_gate=0.499, fs_state_norm=15.956, fs_decay=0.00
- loop 6: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3379
  future_seed: fs_gate=0.499, fs_state_norm=15.956, fs_decay=0.00
- loop 7: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3381
  future_seed: fs_gate=0.499, fs_state_norm=15.956, fs_decay=0.00
- loop 8: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3381
  future_seed: fs_gate=0.499, fs_state_norm=15.956, fs_decay=0.00

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3381; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3381; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3381; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3381; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop5
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop8
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000

## Hole Transfer

- holes24: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3381; indep=0.0000, exact/indep=0.00
- holes16: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3645; indep=0.0000, exact/indep=0.00
- holes32: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3112; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/fs-decay-16x16-24a95d5/runs/loopall16x16-d256-l12-loop8-h32-b16-s600eval-20260608T0545Z-24a95d5/output/futureseed_loop_case_seed52.html
