# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.4498, total=0.4498, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.4477, loop_last_loss=0.4498, sec=403.7
- loop 1: exact=0.0684, valid=0.0684, solved=0.0684, blank_acc=0.7104
  future_seed: fs_gate=0.000, fs_state_norm=0.000
- loop 2: exact=0.0684, valid=0.0684, solved=0.0684, blank_acc=0.7114
  future_seed: fs_gate=0.000, fs_state_norm=0.000
- loop 3: exact=0.0664, valid=0.0664, solved=0.0664, blank_acc=0.7122
  future_seed: fs_gate=0.000, fs_state_norm=0.000
- loop 4: exact=0.0664, valid=0.0664, solved=0.0664, blank_acc=0.7117
  future_seed: fs_gate=0.000, fs_state_norm=0.000
- loop 5: exact=0.0664, valid=0.0664, solved=0.0664, blank_acc=0.7114
  future_seed: fs_gate=0.000, fs_state_norm=0.000

## Eval Noise

- clean eval loop5: exact=0.0664, valid=0.0664, solved=0.0664, blank_acc=0.7114
- noisy eval loop5: exact=0.0664, valid=0.0664, solved=0.0664, blank_acc=0.7085

## Stochastic Rollouts

- K1: oracle_exact=0.0625, oracle_solved=0.0625, disagree=0.0000
  confidence: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.7043; gap=0.0000
  consistency: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.7043; gap=0.0000
  residual: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.7043; gap=0.0000
  majority: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.7043; gap=0.0000
- K4: oracle_exact=0.1406, oracle_solved=0.1406, disagree=0.1302
  confidence: exact=0.0703, valid=0.0703, solved=0.0703, blank_acc=0.7053; gap=0.0703
  consistency: exact=0.0723, valid=0.0723, solved=0.0723, blank_acc=0.7065; gap=0.0684
  residual: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.7043; gap=0.0781
  majority: exact=0.0645, valid=0.0645, solved=0.0645, blank_acc=0.7080; gap=0.0762
- K8: oracle_exact=0.2031, oracle_solved=0.2031, disagree=0.1492
  confidence: exact=0.0723, valid=0.0723, solved=0.0723, blank_acc=0.7092; gap=0.1309
  consistency: exact=0.0762, valid=0.0762, solved=0.0762, blank_acc=0.7070; gap=0.1270
  residual: exact=0.0684, valid=0.0684, solved=0.0684, blank_acc=0.7092; gap=0.1348
  majority: exact=0.0742, valid=0.0742, solved=0.0742, blank_acc=0.7104; gap=0.1289

## Compute Scaling

### loop1
- K1: oracle_exact=0.0645, confidence=0.0645, residual=0.0645, disagree=0.0000
- K4: oracle_exact=0.1152, confidence=0.0703, residual=0.0645, disagree=0.0835
- K8: oracle_exact=0.1465, confidence=0.0645, residual=0.0645, disagree=0.0938
### loop3
- K1: oracle_exact=0.0605, confidence=0.0605, residual=0.0605, disagree=0.0000
- K4: oracle_exact=0.1328, confidence=0.0645, residual=0.0625, disagree=0.1144
- K8: oracle_exact=0.1758, confidence=0.0703, residual=0.0762, disagree=0.1275
### loop5
- K1: oracle_exact=0.0625, confidence=0.0625, residual=0.0625, disagree=0.0000
- K4: oracle_exact=0.1543, confidence=0.0664, residual=0.0586, disagree=0.1309
- K8: oracle_exact=0.2090, confidence=0.0703, residual=0.0449, disagree=0.1481

## Hole Transfer

- holes8: exact=0.0664, valid=0.0664, solved=0.0664, blank_acc=0.7114; indep=0.0656, exact/indep=1.01
- holes12: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.6898; indep=0.0116, exact/indep=1.68
- holes16: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6608; indep=0.0013, exact/indep=0.00

## Decision

Depth loop is not yet stable; later loops reduce full-board exact.

## Artifacts

- case_html: /huyang2/double-loop/runs/9x9-ablate-no-future-seed-20260603T111747Z-5e2253f/output/futureseed_loop_case_seed52.html
