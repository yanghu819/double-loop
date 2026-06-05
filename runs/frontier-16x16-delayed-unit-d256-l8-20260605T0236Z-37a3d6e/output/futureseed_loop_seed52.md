# 16x16 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 16x16, box: 4x4, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0642, total=1.0823, loop_loss=delayed, noise=feature_diff, buffer=8192, loop1_loss=1.0826, loop_last_loss=1.0642, sec=2305.5
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3828
  future_seed: fs_gate=0.498, fs_state_norm=7.965
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3832
  future_seed: fs_gate=0.498, fs_state_norm=7.965
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3839
  future_seed: fs_gate=0.498, fs_state_norm=7.965
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3838
  future_seed: fs_gate=0.498, fs_state_norm=7.965
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3835
  future_seed: fs_gate=0.498, fs_state_norm=7.965
- loop 6: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3832
  future_seed: fs_gate=0.498, fs_state_norm=7.965
- loop 7: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831
  future_seed: fs_gate=0.498, fs_state_norm=7.965
- loop 8: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831
  future_seed: fs_gate=0.498, fs_state_norm=7.965

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831; gap=0.0000
- K4: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831; gap=0.0000
- K8: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K4: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K8: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop4
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K4: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K8: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop8
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K4: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K8: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000

## Hole Transfer

- holes32: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3831; indep=0.0000, exact/indep=0.00
- holes24: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4106; indep=0.0000, exact/indep=0.00
- holes40: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3903; indep=0.0000, exact/indep=0.00
- holes48: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3559; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/frontier-16x16-delayed-unit-d256-l8-20260605T0236Z-37a3d6e/output/futureseed_loop_case_seed52.html
