# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0135, total=0.0135, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.0135, loop_last_loss=0.0135, sec=94.0
- loop 1: exact=0.9922, valid=0.9922, solved=0.9922, blank_acc=0.9985
  future_seed: fs_gate=0.531, fs_state_norm=8.500

## Eval Noise

- clean eval loop1: exact=0.9922, valid=0.9922, solved=0.9922, blank_acc=0.9985
- noisy eval loop1: exact=0.9922, valid=0.9922, solved=0.9922, blank_acc=0.9985

## Stochastic Rollouts

- K1: oracle_exact=0.9902, oracle_solved=0.9902, disagree=0.0000
  confidence: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9983; gap=0.0000
  consistency: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9983; gap=0.0000
  residual: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9983; gap=0.0000
  majority: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9983; gap=0.0000
- K4: oracle_exact=0.9941, oracle_solved=0.9941, disagree=0.0004
  confidence: exact=0.9941, valid=0.9941, solved=0.9941, blank_acc=0.9988; gap=0.0000
  consistency: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9983; gap=0.0039
  residual: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9983; gap=0.0039
  majority: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9983; gap=0.0039
- K8: oracle_exact=0.9941, oracle_solved=0.9941, disagree=0.0004
  confidence: exact=0.9922, valid=0.9922, solved=0.9922, blank_acc=0.9985; gap=0.0020
  consistency: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9983; gap=0.0039
  residual: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9983; gap=0.0039
  majority: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9983; gap=0.0039

## Compute Scaling

### loop1
- K1: oracle_exact=0.9883, confidence=0.9883, residual=0.9883, disagree=0.0000
- K4: oracle_exact=0.9922, confidence=0.9883, residual=0.9883, disagree=0.0004
- K8: oracle_exact=0.9961, confidence=0.9883, residual=0.9883, disagree=0.0005

## Hole Transfer

- holes8: exact=0.9922, valid=0.9922, solved=0.9922, blank_acc=0.9985; indep=0.9883, exact/indep=1.00
- holes12: exact=0.9551, valid=0.9551, solved=0.9551, blank_acc=0.9958; indep=0.9504, exact/indep=1.00
- holes16: exact=0.7715, valid=0.7715, solved=0.7715, blank_acc=0.9810; indep=0.7352, exact/indep=1.05

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/9x9-ablate-no-loop-20260603T111606Z-5e2253f/output/futureseed_loop_case_seed52.html
