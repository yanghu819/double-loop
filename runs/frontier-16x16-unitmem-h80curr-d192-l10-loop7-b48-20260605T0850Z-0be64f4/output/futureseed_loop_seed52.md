# 16x16 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 16x16, box: 4x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0873, total=0.0873, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.4487, loop_last_loss=0.0873, sec=2253.1
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7069
  future_seed: fs_gate=0.510, fs_state_norm=8.161
- loop 2: exact=0.0312, valid=0.0312, solved=0.0312, blank_acc=0.9404
  future_seed: fs_gate=0.510, fs_state_norm=8.161
- loop 3: exact=0.0586, valid=0.0586, solved=0.0586, blank_acc=0.9466
  future_seed: fs_gate=0.510, fs_state_norm=8.161
- loop 4: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.9469
  future_seed: fs_gate=0.510, fs_state_norm=8.161
- loop 5: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.9471
  future_seed: fs_gate=0.510, fs_state_norm=8.161
- loop 6: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.9472
  future_seed: fs_gate=0.510, fs_state_norm=8.161
- loop 7: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.9472
  future_seed: fs_gate=0.510, fs_state_norm=8.161

## Stochastic Rollouts

- K1: oracle_exact=0.0625, oracle_solved=0.0625, disagree=0.0000
  confidence: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.9472; gap=0.0000
  consistency: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.9472; gap=0.0000
  residual: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.9472; gap=0.0000
  majority: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.9472; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0586, confidence=0.0586, residual=0.0586, disagree=0.0000
### loop5
- K1: oracle_exact=0.0625, confidence=0.0625, residual=0.0625, disagree=0.0000
### loop7
- K1: oracle_exact=0.0625, confidence=0.0625, residual=0.0625, disagree=0.0000

## Hole Transfer

- holes80: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.9472; indep=0.0130, exact/indep=4.81
- holes64: exact=0.4297, valid=0.4297, solved=0.4297, blank_acc=0.9832; indep=0.3371, exact/indep=1.27
- holes72: exact=0.2031, valid=0.2031, solved=0.2031, blank_acc=0.9688; indep=0.1017, exact/indep=2.00
- holes88: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.9107; indep=0.0003, exact/indep=14.70
- holes96: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.8712; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/h80mem-0be64f4/runs/frontier-16x16-unitmem-h80curr-d192-l10-loop7-b48-20260605T0850Z-0be64f4/output/futureseed_loop_case_seed52.html
