# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0064, total=0.0064, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.0280, loop_last_loss=0.0064, sec=422.0
- loop 1: exact=0.9629, valid=0.9629, solved=0.9629, blank_acc=0.9954
  future_seed: fs_gate=0.534, fs_state_norm=8.538
- loop 2: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000
  future_seed: fs_gate=0.534, fs_state_norm=8.538
- loop 3: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000
  future_seed: fs_gate=0.534, fs_state_norm=8.538
- loop 4: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000
  future_seed: fs_gate=0.534, fs_state_norm=8.538
- loop 5: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000
  future_seed: fs_gate=0.534, fs_state_norm=8.538

## Eval Noise

- clean eval loop5: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000
- noisy eval loop5: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000

## Stochastic Rollouts

- K1: oracle_exact=0.9961, oracle_solved=0.9961, disagree=0.0000
  confidence: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9995; gap=0.0000
  consistency: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9995; gap=0.0000
  residual: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9995; gap=0.0000
  majority: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9995; gap=0.0000
- K4: oracle_exact=1.0000, oracle_solved=1.0000, disagree=0.0003
  confidence: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000; gap=0.0000
  consistency: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000; gap=0.0000
  residual: exact=0.9980, valid=0.9980, solved=0.9980, blank_acc=0.9998; gap=0.0020
  majority: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000; gap=0.0000
- K8: oracle_exact=1.0000, oracle_solved=1.0000, disagree=0.0003
  confidence: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000; gap=0.0000
  consistency: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000; gap=0.0000
  residual: exact=0.9980, valid=0.9980, solved=0.9980, blank_acc=0.9998; gap=0.0020
  majority: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.9473, confidence=0.9473, residual=0.9473, disagree=0.0000
- K4: oracle_exact=0.9805, confidence=0.9668, residual=0.9473, disagree=0.0027
- K8: oracle_exact=0.9824, confidence=0.9648, residual=0.9473, disagree=0.0030
### loop3
- K1: oracle_exact=0.9980, confidence=0.9980, residual=0.9980, disagree=0.0000
- K4: oracle_exact=1.0000, confidence=1.0000, residual=1.0000, disagree=0.0002
- K8: oracle_exact=1.0000, confidence=1.0000, residual=1.0000, disagree=0.0003
### loop5
- K1: oracle_exact=0.9980, confidence=0.9980, residual=0.9980, disagree=0.0000
- K4: oracle_exact=1.0000, confidence=1.0000, residual=0.9980, disagree=0.0004
- K8: oracle_exact=1.0000, confidence=1.0000, residual=1.0000, disagree=0.0006

## Hole Transfer

- holes8: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000; indep=1.0000, exact/indep=1.00
- holes12: exact=0.9629, valid=0.9629, solved=0.9629, blank_acc=0.9966; indep=0.9597, exact/indep=1.00
- holes16: exact=0.8535, valid=0.8535, solved=0.8535, blank_acc=0.9883; indep=0.8281, exact/indep=1.03

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/9x9-cliff-cuda-20260603T103352Z-92ee7b9/output/futureseed_loop_case_seed52.html
