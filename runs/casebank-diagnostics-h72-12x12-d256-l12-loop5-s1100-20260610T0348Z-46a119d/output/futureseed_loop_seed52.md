# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.1525, total=0.1525, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop1_loss=0.8601, loop_last_loss=0.1525, sec=1157.0
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7654
  future_seed: fs_gate=0.542, fs_update=1.000, fs_state_norm=8.673, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 2: exact=0.2266, valid=0.2344, solved=0.2344, blank_acc=0.9602
  future_seed: fs_gate=0.542, fs_update=1.000, fs_state_norm=8.673, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 3: exact=0.5911, valid=0.6198, solved=0.6198, blank_acc=0.9809
  future_seed: fs_gate=0.542, fs_update=1.000, fs_state_norm=8.673, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 4: exact=0.6354, valid=0.6615, solved=0.6615, blank_acc=0.9829
  future_seed: fs_gate=0.542, fs_update=1.000, fs_state_norm=8.673, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 5: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833
  future_seed: fs_gate=0.542, fs_update=1.000, fs_state_norm=8.673, fs_decay=0.00, fb_in=0.000, fb_next=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.6484, oracle_solved=0.6745, disagree=0.0000
  confidence: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833; gap=0.0000
  consistency: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833; gap=0.0000
  residual: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833; gap=0.0000
  majority: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833; gap=0.0000
- K4: oracle_exact=0.6484, oracle_solved=0.6745, disagree=0.0000
  confidence: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833; gap=0.0000
  consistency: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833; gap=0.0000
  residual: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833; gap=0.0000
  majority: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833; gap=0.0000
- K8: oracle_exact=0.6484, oracle_solved=0.6745, disagree=0.0000
  confidence: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833; gap=0.0000
  consistency: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833; gap=0.0000
  residual: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833; gap=0.0000
  majority: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K4: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K8: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.5911, confidence=0.5911, residual=0.5911, disagree=0.0000
- K4: oracle_exact=0.5911, confidence=0.5911, residual=0.5911, disagree=0.0000
- K8: oracle_exact=0.5911, confidence=0.5911, residual=0.5911, disagree=0.0000
### loop5
- K1: oracle_exact=0.6484, confidence=0.6484, residual=0.6484, disagree=0.0000
- K4: oracle_exact=0.6484, confidence=0.6484, residual=0.6484, disagree=0.0000
- K8: oracle_exact=0.6484, confidence=0.6484, residual=0.6484, disagree=0.0000

## Case Bank

- h72: index=/huyang2/double-loop/.worktrees/casebank-diagnostics-20260610T033611Z-46a119d/runs/casebank-diagnostics-h72-12x12-d256-l12-loop5-s1100-20260610T0348Z-46a119d/output/case_bank/h72/index.html; cases=/huyang2/double-loop/.worktrees/casebank-diagnostics-20260610T033611Z-46a119d/runs/casebank-diagnostics-h72-12x12-d256-l12-loop5-s1100-20260610T0348Z-46a119d/output/case_bank/h72/cases.json; final_loop=5, exact=0.0742, blank_acc=0.9023, selected={'solved_by_loop': 8, 'almost_solved': 8, 'hard_failure': 8}

## Hole Transfer

- holes60: exact=0.6484, valid=0.6745, solved=0.6745, blank_acc=0.9833; indep=0.3638, exact/indep=1.78
- holes48: exact=0.9531, valid=0.9583, solved=0.9583, blank_acc=0.9975; indep=0.8870, exact/indep=1.07
- holes72: exact=0.0677, valid=0.0755, solved=0.0755, blank_acc=0.8980; indep=0.0004, exact/indep=156.09
- holes84: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7026; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/casebank-diagnostics-20260610T033611Z-46a119d/runs/casebank-diagnostics-h72-12x12-d256-l12-loop5-s1100-20260610T0348Z-46a119d/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/casebank-diagnostics-20260610T033611Z-46a119d/runs/casebank-diagnostics-h72-12x12-d256-l12-loop5-s1100-20260610T0348Z-46a119d/output/case_bank
