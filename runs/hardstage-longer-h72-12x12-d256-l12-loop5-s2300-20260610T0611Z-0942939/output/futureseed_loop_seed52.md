# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0758, total=0.0758, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop1_loss=0.9457, loop_last_loss=0.0758, sec=2384.5
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7953
  future_seed: fs_gate=0.561, fs_update=1.000, fs_state_norm=8.973, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 2: exact=0.4010, valid=0.4036, solved=0.4036, blank_acc=0.9730
  future_seed: fs_gate=0.561, fs_update=1.000, fs_state_norm=8.973, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 3: exact=0.8490, valid=0.8828, solved=0.8828, blank_acc=0.9906
  future_seed: fs_gate=0.561, fs_update=1.000, fs_state_norm=8.973, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 4: exact=0.8750, valid=0.9115, solved=0.9115, blank_acc=0.9911
  future_seed: fs_gate=0.561, fs_update=1.000, fs_state_norm=8.973, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 5: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911
  future_seed: fs_gate=0.561, fs_update=1.000, fs_state_norm=8.973, fs_decay=0.00, fb_in=0.000, fb_next=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.8750, oracle_solved=0.9141, disagree=0.0000
  confidence: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911; gap=0.0000
  consistency: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911; gap=0.0000
  residual: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911; gap=0.0000
  majority: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911; gap=0.0000
- K4: oracle_exact=0.8750, oracle_solved=0.9141, disagree=0.0000
  confidence: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911; gap=0.0000
  consistency: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911; gap=0.0000
  residual: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911; gap=0.0000
  majority: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911; gap=0.0000
- K8: oracle_exact=0.8750, oracle_solved=0.9141, disagree=0.0000
  confidence: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911; gap=0.0000
  consistency: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911; gap=0.0000
  residual: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911; gap=0.0000
  majority: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K4: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K8: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.8490, confidence=0.8490, residual=0.8490, disagree=0.0000
- K4: oracle_exact=0.8490, confidence=0.8490, residual=0.8490, disagree=0.0000
- K8: oracle_exact=0.8490, confidence=0.8490, residual=0.8490, disagree=0.0000
### loop5
- K1: oracle_exact=0.8750, confidence=0.8750, residual=0.8750, disagree=0.0000
- K4: oracle_exact=0.8750, confidence=0.8750, residual=0.8750, disagree=0.0000
- K8: oracle_exact=0.8750, confidence=0.8750, residual=0.8750, disagree=0.0000

## Case Bank

- h72: index=/huyang2/double-loop/.worktrees/hardstage2300-h72-20260610T061045Z-0942939/runs/hardstage-longer-h72-12x12-d256-l12-loop5-s2300-20260610T0611Z-0942939/output/case_bank/h72/index.html; cases=/huyang2/double-loop/.worktrees/hardstage2300-h72-20260610T061045Z-0942939/runs/hardstage-longer-h72-12x12-d256-l12-loop5-s2300-20260610T0611Z-0942939/output/case_bank/h72/cases.json; final_loop=5, exact=0.3955, blank_acc=0.9534, selected={'solved_by_loop': 8, 'almost_solved': 8, 'hard_failure': 8}

## Hole Transfer

- holes60: exact=0.8750, valid=0.9141, solved=0.9141, blank_acc=0.9911; indep=0.5865, exact/indep=1.49
- holes48: exact=0.9818, valid=0.9844, solved=0.9844, blank_acc=0.9988; indep=0.9443, exact/indep=1.04
- holes72: exact=0.4010, valid=0.4453, solved=0.4453, blank_acc=0.9510; indep=0.0268, exact/indep=14.94
- holes84: exact=0.0078, valid=0.0104, solved=0.0104, blank_acc=0.7982; indep=0.0000, exact/indep=1299025.70

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/hardstage2300-h72-20260610T061045Z-0942939/runs/hardstage-longer-h72-12x12-d256-l12-loop5-s2300-20260610T0611Z-0942939/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/hardstage2300-h72-20260610T061045Z-0942939/runs/hardstage-longer-h72-12x12-d256-l12-loop5-s2300-20260610T0611Z-0942939/output/case_bank
