# 6x6 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 6x6, box: 2x3, hole pattern: `unit`.

## Loop Metrics

- train: ce=0.0038, total=0.0038, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.0040, loop_last_loss=0.0038, sec=2576.6
- loop 1: exact=0.9883, valid=0.9883, solved=0.9883, blank_acc=0.9971
  future_seed: fs_gate=0.523, fs_state_norm=6.278
- loop 2: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9973
  future_seed: fs_gate=0.523, fs_state_norm=6.278
- loop 3: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9976
  future_seed: fs_gate=0.523, fs_state_norm=6.278

## Eval Noise

- clean eval loop3: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9976
- noisy eval loop3: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9976

## Stochastic Rollouts

- K1: oracle_exact=0.9873, oracle_solved=0.9873, disagree=0.0000
  confidence: exact=0.9873, valid=0.9873, solved=0.9873, blank_acc=0.9971; gap=0.0000
  consistency: exact=0.9873, valid=0.9873, solved=0.9873, blank_acc=0.9971; gap=0.0000
  residual: exact=0.9873, valid=0.9873, solved=0.9873, blank_acc=0.9971; gap=0.0000
  majority: exact=0.9873, valid=0.9873, solved=0.9873, blank_acc=0.9971; gap=0.0000
- K4: oracle_exact=0.9912, oracle_solved=0.9912, disagree=0.0005
  confidence: exact=0.9893, valid=0.9893, solved=0.9893, blank_acc=0.9973; gap=0.0020
  consistency: exact=0.9883, valid=0.9883, solved=0.9883, blank_acc=0.9973; gap=0.0029
  residual: exact=0.9883, valid=0.9883, solved=0.9883, blank_acc=0.9971; gap=0.0029
  majority: exact=0.9893, valid=0.9893, solved=0.9893, blank_acc=0.9973; gap=0.0020
- K8: oracle_exact=0.9941, oracle_solved=0.9941, disagree=0.0007
  confidence: exact=0.9893, valid=0.9893, solved=0.9893, blank_acc=0.9971; gap=0.0049
  consistency: exact=0.9893, valid=0.9893, solved=0.9893, blank_acc=0.9973; gap=0.0049
  residual: exact=0.9883, valid=0.9883, solved=0.9883, blank_acc=0.9966; gap=0.0059
  majority: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9973; gap=0.0039
- K16: oracle_exact=0.9951, oracle_solved=0.9951, disagree=0.0007
  confidence: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9973; gap=0.0049
  consistency: exact=0.9893, valid=0.9893, solved=0.9893, blank_acc=0.9973; gap=0.0059
  residual: exact=0.9912, valid=0.9912, solved=0.9912, blank_acc=0.9976; gap=0.0039
  majority: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9973; gap=0.0049

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/scale-window-20260603T064353Z-eaf6b40/output/futureseed_loop_case_seed52.html
