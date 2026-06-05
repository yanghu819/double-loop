# 16x16 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 16x16, box: 4x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.1226, total=0.1226, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.5101, loop_last_loss=0.1226, sec=1855.8
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.8027
  future_seed: fs_gate=0.505, fs_state_norm=8.084
- loop 2: exact=0.2188, valid=0.2227, solved=0.2227, blank_acc=0.9669
  future_seed: fs_gate=0.505, fs_state_norm=8.084
- loop 3: exact=0.2344, valid=0.2344, solved=0.2344, blank_acc=0.9683
  future_seed: fs_gate=0.505, fs_state_norm=8.084
- loop 4: exact=0.2305, valid=0.2305, solved=0.2305, blank_acc=0.9685
  future_seed: fs_gate=0.505, fs_state_norm=8.084
- loop 5: exact=0.2305, valid=0.2305, solved=0.2305, blank_acc=0.9684
  future_seed: fs_gate=0.505, fs_state_norm=8.084
- loop 6: exact=0.2305, valid=0.2305, solved=0.2305, blank_acc=0.9683
  future_seed: fs_gate=0.505, fs_state_norm=8.084
- loop 7: exact=0.2305, valid=0.2305, solved=0.2305, blank_acc=0.9684
  future_seed: fs_gate=0.505, fs_state_norm=8.084

## Stochastic Rollouts

- K1: oracle_exact=0.2305, oracle_solved=0.2305, disagree=0.0000
  confidence: exact=0.2305, valid=0.2305, solved=0.2305, blank_acc=0.9684; gap=0.0000
  consistency: exact=0.2305, valid=0.2305, solved=0.2305, blank_acc=0.9684; gap=0.0000
  residual: exact=0.2305, valid=0.2305, solved=0.2305, blank_acc=0.9684; gap=0.0000
  majority: exact=0.2305, valid=0.2305, solved=0.2305, blank_acc=0.9684; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.2344, confidence=0.2344, residual=0.2344, disagree=0.0000
### loop5
- K1: oracle_exact=0.2305, confidence=0.2305, residual=0.2305, disagree=0.0000
### loop7
- K1: oracle_exact=0.2305, confidence=0.2305, residual=0.2305, disagree=0.0000

## Hole Transfer

- holes64: exact=0.2305, valid=0.2305, solved=0.2305, blank_acc=0.9684; indep=0.1285, exact/indep=1.79
- holes48: exact=0.6992, valid=0.7070, solved=0.7070, blank_acc=0.9896; indep=0.6049, exact/indep=1.16
- holes80: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.9174; indep=0.0010, exact/indep=0.00
- holes96: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.8343; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/unitmem-79004cf/runs/frontier-16x16-unitmem-d192-l10-loop7-h80-b48-20260605T0803Z-79004cf/output/futureseed_loop_case_seed52.html
