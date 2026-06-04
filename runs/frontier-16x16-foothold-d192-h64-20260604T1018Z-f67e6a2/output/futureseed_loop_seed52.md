# 16x16 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 16x16, box: 4x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.5792, total=0.5792, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.9110, loop_last_loss=0.5792, sec=1353.9
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6528
  future_seed: fs_gate=0.515, fs_state_norm=8.240
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7304
  future_seed: fs_gate=0.515, fs_state_norm=8.240
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7374
  future_seed: fs_gate=0.515, fs_state_norm=8.240
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7398
  future_seed: fs_gate=0.515, fs_state_norm=8.240
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7393
  future_seed: fs_gate=0.515, fs_state_norm=8.240

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7393; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7393; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7393; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7393; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop5
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000

## Hole Transfer

- holes48: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7393; indep=0.0000, exact/indep=0.00
- holes32: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.8287; indep=0.0025, exact/indep=1.59
- holes64: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6552; indep=0.0000, exact/indep=0.00
- holes80: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5704; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/frontier-16x16-foothold-d192-h64-20260604T1018Z-f67e6a2/output/futureseed_loop_case_seed52.html
