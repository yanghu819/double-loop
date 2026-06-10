# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.1621, total=0.1625, loop_loss=delayed, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop1_loss=0.7776, loop_last_loss=0.1621, sec=1460.5
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.8020
  future_seed: fs_gate=0.528, fs_update=1.000, fs_state_norm=8.446, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 2: exact=0.2318, valid=0.2422, solved=0.2422, blank_acc=0.9556
  future_seed: fs_gate=0.528, fs_update=1.000, fs_state_norm=8.446, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 3: exact=0.4036, valid=0.4167, solved=0.4167, blank_acc=0.9664
  future_seed: fs_gate=0.528, fs_update=1.000, fs_state_norm=8.446, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 4: exact=0.4167, valid=0.4297, solved=0.4297, blank_acc=0.9671
  future_seed: fs_gate=0.528, fs_update=1.000, fs_state_norm=8.446, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 5: exact=0.4167, valid=0.4323, solved=0.4323, blank_acc=0.9674
  future_seed: fs_gate=0.528, fs_update=1.000, fs_state_norm=8.446, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 6: exact=0.4219, valid=0.4375, solved=0.4375, blank_acc=0.9676
  future_seed: fs_gate=0.528, fs_update=1.000, fs_state_norm=8.446, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 7: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676
  future_seed: fs_gate=0.528, fs_update=1.000, fs_state_norm=8.446, fs_decay=0.00, fb_in=0.000, fb_next=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.4245, oracle_solved=0.4401, disagree=0.0000
  confidence: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676; gap=0.0000
  consistency: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676; gap=0.0000
  residual: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676; gap=0.0000
  majority: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676; gap=0.0000
- K4: oracle_exact=0.4245, oracle_solved=0.4401, disagree=0.0000
  confidence: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676; gap=0.0000
  consistency: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676; gap=0.0000
  residual: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676; gap=0.0000
  majority: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676; gap=0.0000
- K8: oracle_exact=0.4245, oracle_solved=0.4401, disagree=0.0000
  confidence: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676; gap=0.0000
  consistency: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676; gap=0.0000
  residual: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676; gap=0.0000
  majority: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K4: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K8: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.4036, confidence=0.4036, residual=0.4036, disagree=0.0000
- K4: oracle_exact=0.4036, confidence=0.4036, residual=0.4036, disagree=0.0000
- K8: oracle_exact=0.4036, confidence=0.4036, residual=0.4036, disagree=0.0000
### loop5
- K1: oracle_exact=0.4167, confidence=0.4167, residual=0.4167, disagree=0.0000
- K4: oracle_exact=0.4167, confidence=0.4167, residual=0.4167, disagree=0.0000
- K8: oracle_exact=0.4167, confidence=0.4167, residual=0.4167, disagree=0.0000
### loop7
- K1: oracle_exact=0.4245, confidence=0.4245, residual=0.4245, disagree=0.0000
- K4: oracle_exact=0.4245, confidence=0.4245, residual=0.4245, disagree=0.0000
- K8: oracle_exact=0.4245, confidence=0.4245, residual=0.4245, disagree=0.0000

## Case Bank

- h72: index=/huyang2/double-loop/.worktrees/loopcredit-h72-20260610T041946Z-0942939/runs/loopcredit-delayed-h72-12x12-d256-l12-loop7-s1100-20260610T0420Z-0942939/output/case_bank/h72/index.html; cases=/huyang2/double-loop/.worktrees/loopcredit-h72-20260610T041946Z-0942939/runs/loopcredit-delayed-h72-12x12-d256-l12-loop7-s1100-20260610T0420Z-0942939/output/case_bank/h72/cases.json; final_loop=7, exact=0.0283, blank_acc=0.8860, selected={'solved_by_loop': 8, 'almost_solved': 8, 'hard_failure': 8}

## Hole Transfer

- holes60: exact=0.4245, valid=0.4401, solved=0.4401, blank_acc=0.9676; indep=0.1388, exact/indep=3.06
- holes48: exact=0.8464, valid=0.8646, solved=0.8646, blank_acc=0.9913; indep=0.6580, exact/indep=1.29
- holes72: exact=0.0156, valid=0.0208, solved=0.0208, blank_acc=0.8793; indep=0.0001, exact/indep=163.88
- holes84: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6939; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/loopcredit-h72-20260610T041946Z-0942939/runs/loopcredit-delayed-h72-12x12-d256-l12-loop7-s1100-20260610T0420Z-0942939/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/loopcredit-h72-20260610T041946Z-0942939/runs/loopcredit-delayed-h72-12x12-d256-l12-loop7-s1100-20260610T0420Z-0942939/output/case_bank
