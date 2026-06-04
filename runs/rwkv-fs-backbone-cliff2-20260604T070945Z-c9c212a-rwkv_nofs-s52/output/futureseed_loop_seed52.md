# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.6316, total=0.6316, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.6479, loop_last_loss=0.6316, sec=829.7
- loop 1: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.6931
  future_seed: fs_gate=0.000, fs_state_norm=0.000
- loop 2: exact=0.0059, valid=0.0059, solved=0.0059, blank_acc=0.6963
  future_seed: fs_gate=0.000, fs_state_norm=0.000
- loop 3: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6946
  future_seed: fs_gate=0.000, fs_state_norm=0.000
- loop 4: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954
  future_seed: fs_gate=0.000, fs_state_norm=0.000
- loop 5: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954
  future_seed: fs_gate=0.000, fs_state_norm=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0039, oracle_solved=0.0039, disagree=0.0000
  confidence: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954; gap=0.0000
  consistency: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954; gap=0.0000
  residual: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954; gap=0.0000
  majority: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954; gap=0.0000
- K4: oracle_exact=0.0039, oracle_solved=0.0039, disagree=0.0000
  confidence: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954; gap=0.0000
  consistency: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954; gap=0.0000
  residual: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954; gap=0.0000
  majority: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954; gap=0.0000
- K8: oracle_exact=0.0039, oracle_solved=0.0039, disagree=0.0000
  confidence: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954; gap=0.0000
  consistency: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954; gap=0.0000
  residual: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954; gap=0.0000
  majority: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0078, confidence=0.0078, residual=0.0078, disagree=0.0000
- K4: oracle_exact=0.0078, confidence=0.0078, residual=0.0078, disagree=0.0000
- K8: oracle_exact=0.0078, confidence=0.0078, residual=0.0078, disagree=0.0000
### loop3
- K1: oracle_exact=0.0039, confidence=0.0039, residual=0.0039, disagree=0.0000
- K4: oracle_exact=0.0039, confidence=0.0039, residual=0.0039, disagree=0.0000
- K8: oracle_exact=0.0039, confidence=0.0039, residual=0.0039, disagree=0.0000
### loop5
- K1: oracle_exact=0.0039, confidence=0.0039, residual=0.0039, disagree=0.0000
- K4: oracle_exact=0.0039, confidence=0.0039, residual=0.0039, disagree=0.0000
- K8: oracle_exact=0.0039, confidence=0.0039, residual=0.0039, disagree=0.0000

## Hole Transfer

- holes16: exact=0.0039, valid=0.0039, solved=0.0039, blank_acc=0.6954; indep=0.0030, exact/indep=1.31
- holes20: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6669; indep=0.0003, exact/indep=0.00
- holes24: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6350; indep=0.0000, exact/indep=0.00
- holes28: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6004; indep=0.0000, exact/indep=0.00
- holes32: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5493; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is not yet stable; later loops reduce full-board exact.

## Artifacts

- case_html: /huyang2/double-loop/runs/rwkv-fs-backbone-cliff2-20260604T070945Z-c9c212a-rwkv_nofs-s52/output/futureseed_loop_case_seed52.html
