# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.8823, total=0.8823, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.8797, loop_last_loss=0.8823, sec=8.9
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3682
  future_seed: fs_gate=0.499, fs_state_norm=7.983
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3701
  future_seed: fs_gate=0.499, fs_state_norm=7.983

## Eval Noise

- clean eval loop2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3701
- noisy eval loop2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3691

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3682; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3682; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3682; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3682; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop2
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000

## Hole Transfer

- holes8: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3701; indep=0.0004, exact/indep=0.00
- holes12: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3099; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/9x9-rwkv7-cuda-throughput-20260603T103019Z-2c21f96/output/futureseed_loop_case_seed52.html
