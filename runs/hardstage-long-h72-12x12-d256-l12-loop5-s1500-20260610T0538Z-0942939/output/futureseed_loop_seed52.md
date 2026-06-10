# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0802, total=0.0802, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop1_loss=0.8825, loop_last_loss=0.0802, sec=1567.4
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7599
  future_seed: fs_gate=0.550, fs_update=1.000, fs_state_norm=8.801, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 2: exact=0.3359, valid=0.3385, solved=0.3385, blank_acc=0.9682
  future_seed: fs_gate=0.550, fs_update=1.000, fs_state_norm=8.801, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 3: exact=0.7266, valid=0.7448, solved=0.7448, blank_acc=0.9872
  future_seed: fs_gate=0.550, fs_update=1.000, fs_state_norm=8.801, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 4: exact=0.7708, valid=0.7891, solved=0.7891, blank_acc=0.9882
  future_seed: fs_gate=0.550, fs_update=1.000, fs_state_norm=8.801, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 5: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884
  future_seed: fs_gate=0.550, fs_update=1.000, fs_state_norm=8.801, fs_decay=0.00, fb_in=0.000, fb_next=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.7786, oracle_solved=0.7995, disagree=0.0000
  confidence: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884; gap=0.0000
  consistency: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884; gap=0.0000
  residual: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884; gap=0.0000
  majority: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884; gap=0.0000
- K4: oracle_exact=0.7786, oracle_solved=0.7995, disagree=0.0000
  confidence: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884; gap=0.0000
  consistency: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884; gap=0.0000
  residual: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884; gap=0.0000
  majority: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884; gap=0.0000
- K8: oracle_exact=0.7786, oracle_solved=0.7995, disagree=0.0000
  confidence: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884; gap=0.0000
  consistency: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884; gap=0.0000
  residual: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884; gap=0.0000
  majority: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K4: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K8: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.7266, confidence=0.7266, residual=0.7266, disagree=0.0000
- K4: oracle_exact=0.7266, confidence=0.7266, residual=0.7266, disagree=0.0000
- K8: oracle_exact=0.7266, confidence=0.7266, residual=0.7266, disagree=0.0000
### loop5
- K1: oracle_exact=0.7786, confidence=0.7786, residual=0.7786, disagree=0.0000
- K4: oracle_exact=0.7786, confidence=0.7786, residual=0.7786, disagree=0.0000
- K8: oracle_exact=0.7786, confidence=0.7786, residual=0.7786, disagree=0.0000

## Case Bank

- h72: index=/huyang2/double-loop/.worktrees/hardstage-h72-20260610T053842Z-0942939/runs/hardstage-long-h72-12x12-d256-l12-loop5-s1500-20260610T0538Z-0942939/output/case_bank/h72/index.html; cases=/huyang2/double-loop/.worktrees/hardstage-h72-20260610T053842Z-0942939/runs/hardstage-long-h72-12x12-d256-l12-loop5-s1500-20260610T0538Z-0942939/output/case_bank/h72/cases.json; final_loop=5, exact=0.1865, blank_acc=0.9293, selected={'solved_by_loop': 8, 'almost_solved': 8, 'hard_failure': 8}

## Hole Transfer

- holes60: exact=0.7786, valid=0.7995, solved=0.7995, blank_acc=0.9884; indep=0.4969, exact/indep=1.57
- holes48: exact=0.9635, valid=0.9714, solved=0.9714, blank_acc=0.9970; indep=0.8641, exact/indep=1.12
- holes72: exact=0.1745, valid=0.1849, solved=0.1849, blank_acc=0.9251; indep=0.0037, exact/indep=47.33
- holes84: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7378; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/hardstage-h72-20260610T053842Z-0942939/runs/hardstage-long-h72-12x12-d256-l12-loop5-s1500-20260610T0538Z-0942939/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/hardstage-h72-20260610T053842Z-0942939/runs/hardstage-long-h72-12x12-d256-l12-loop5-s1500-20260610T0538Z-0942939/output/case_bank
