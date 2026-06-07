# 25x25 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 25x25, box: 5x5, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0715, total=0.0715, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.1106, loop_last_loss=0.0715, sec=1216.8
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.8880
  future_seed: fs_gate=0.494, fs_state_norm=7.906
- loop 2: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.9351
  future_seed: fs_gate=0.494, fs_state_norm=7.906
- loop 3: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.9377
  future_seed: fs_gate=0.494, fs_state_norm=7.906
- loop 4: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.9379
  future_seed: fs_gate=0.494, fs_state_norm=7.906
- loop 5: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.9381
  future_seed: fs_gate=0.494, fs_state_norm=7.906

## Stochastic Rollouts

- K1: oracle_exact=0.0078, oracle_solved=0.0078, disagree=0.0000
  confidence: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.9381; gap=0.0000
  consistency: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.9381; gap=0.0000
  residual: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.9381; gap=0.0000
  majority: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.9381; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0078, confidence=0.0078, residual=0.0078, disagree=0.0000
### loop5
- K1: oracle_exact=0.0078, confidence=0.0078, residual=0.0078, disagree=0.0000

## Hole Transfer

- holes100: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.9381; indep=0.0017, exact/indep=4.66
- holes75: exact=0.1289, valid=0.1289, solved=0.1289, blank_acc=0.9714; indep=0.1131, exact/indep=1.14
- holes90: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.9510; indep=0.0109, exact/indep=0.72
- holes110: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.9230; indep=0.0001, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/upperbound-25x25-h100-46a0b06/runs/upperbound-25x25-unitmem-h100curr-d160-l8-loop5-b32-20260607T050029Z-46a0b06/output/futureseed_loop_case_seed52.html
