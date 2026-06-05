# 25x25 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 25x25, box: 5x5, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0444, total=0.0444, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.0601, loop_last_loss=0.0444, sec=997.3
- loop 1: exact=0.0117, valid=0.0117, solved=0.0117, blank_acc=0.9370
  future_seed: fs_gate=0.489, fs_state_norm=7.830
- loop 2: exact=0.0508, valid=0.0508, solved=0.0508, blank_acc=0.9593
  future_seed: fs_gate=0.489, fs_state_norm=7.830
- loop 3: exact=0.0586, valid=0.0586, solved=0.0586, blank_acc=0.9608
  future_seed: fs_gate=0.489, fs_state_norm=7.830
- loop 4: exact=0.0742, valid=0.0742, solved=0.0742, blank_acc=0.9610
  future_seed: fs_gate=0.489, fs_state_norm=7.830
- loop 5: exact=0.0742, valid=0.0742, solved=0.0742, blank_acc=0.9611
  future_seed: fs_gate=0.489, fs_state_norm=7.830

## Stochastic Rollouts

- K1: oracle_exact=0.0742, oracle_solved=0.0742, disagree=0.0000
  confidence: exact=0.0742, valid=0.0742, solved=0.0742, blank_acc=0.9611; gap=0.0000
  consistency: exact=0.0742, valid=0.0742, solved=0.0742, blank_acc=0.9611; gap=0.0000
  residual: exact=0.0742, valid=0.0742, solved=0.0742, blank_acc=0.9611; gap=0.0000
  majority: exact=0.0742, valid=0.0742, solved=0.0742, blank_acc=0.9611; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0117, confidence=0.0117, residual=0.0117, disagree=0.0000
### loop3
- K1: oracle_exact=0.0586, confidence=0.0586, residual=0.0586, disagree=0.0000
### loop5
- K1: oracle_exact=0.0742, confidence=0.0742, residual=0.0742, disagree=0.0000

## Hole Transfer

- holes75: exact=0.0742, valid=0.0742, solved=0.0742, blank_acc=0.9611; indep=0.0510, exact/indep=1.46
- holes50: exact=0.5156, valid=0.5156, solved=0.5156, blank_acc=0.9867; indep=0.5125, exact/indep=1.01
- holes100: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.9224; indep=0.0003, exact/indep=0.00
- holes125: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.8703; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/feas25-710bb46/runs/frontier-25x25-unitmem-feas-d160-l8-loop5-b32-20260605T0937Z-710bb46/output/futureseed_loop_case_seed52.html
