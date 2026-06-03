# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0007, total=0.0007, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.0146, loop_last_loss=0.0007, sec=424.4
- loop 1: exact=0.9570, valid=0.9570, solved=0.9570, blank_acc=0.9944
  future_seed: fs_gate=0.528, fs_state_norm=8.441
- loop 2: exact=0.9980, valid=0.9980, solved=0.9980, blank_acc=0.9998
  future_seed: fs_gate=0.528, fs_state_norm=8.441
- loop 3: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000
  future_seed: fs_gate=0.528, fs_state_norm=8.441
- loop 4: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000
  future_seed: fs_gate=0.528, fs_state_norm=8.441
- loop 5: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000
  future_seed: fs_gate=0.528, fs_state_norm=8.441

## Stochastic Rollouts

- K1: oracle_exact=0.9980, oracle_solved=0.9980, disagree=0.0000
  confidence: exact=0.9980, valid=0.9980, solved=0.9980, blank_acc=0.9998; gap=0.0000
  consistency: exact=0.9980, valid=0.9980, solved=0.9980, blank_acc=0.9998; gap=0.0000
  residual: exact=0.9980, valid=0.9980, solved=0.9980, blank_acc=0.9998; gap=0.0000
  majority: exact=0.9980, valid=0.9980, solved=0.9980, blank_acc=0.9998; gap=0.0000
- K4: oracle_exact=1.0000, oracle_solved=1.0000, disagree=0.0002
  confidence: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000; gap=0.0000
  consistency: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000; gap=0.0000
  residual: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000; gap=0.0000
  majority: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000; gap=0.0000
- K8: oracle_exact=1.0000, oracle_solved=1.0000, disagree=0.0002
  confidence: exact=0.9980, valid=0.9980, solved=0.9980, blank_acc=0.9995; gap=0.0020
  consistency: exact=0.9980, valid=0.9980, solved=0.9980, blank_acc=0.9998; gap=0.0020
  residual: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000; gap=0.0000
  majority: exact=0.9980, valid=0.9980, solved=0.9980, blank_acc=0.9998; gap=0.0020

## Compute Scaling

### loop1
- K1: oracle_exact=0.9492, confidence=0.9492, residual=0.9492, disagree=0.0000
- K4: oracle_exact=0.9805, confidence=0.9570, residual=0.9492, disagree=0.0020
- K8: oracle_exact=0.9844, confidence=0.9590, residual=0.9492, disagree=0.0022
### loop3
- K1: oracle_exact=0.9980, confidence=0.9980, residual=0.9980, disagree=0.0000
- K4: oracle_exact=1.0000, confidence=0.9980, residual=0.9961, disagree=0.0002
- K8: oracle_exact=1.0000, confidence=0.9980, residual=0.9980, disagree=0.0002
### loop5
- K1: oracle_exact=0.9961, confidence=0.9961, residual=0.9961, disagree=0.0000
- K4: oracle_exact=0.9980, confidence=0.9980, residual=0.9980, disagree=0.0002
- K8: oracle_exact=1.0000, confidence=1.0000, residual=0.9980, disagree=0.0002

## Hole Transfer

- holes8: exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000; indep=1.0000, exact/indep=1.00
- holes12: exact=0.9766, valid=0.9766, solved=0.9766, blank_acc=0.9972; indep=0.9673, exact/indep=1.01
- holes16: exact=0.9004, valid=0.9004, solved=0.9004, blank_acc=0.9923; indep=0.8838, exact/indep=1.02

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/9x9-ablate-no-feature-noise-20260603T112449Z-5e2253f/output/futureseed_loop_case_seed52.html
