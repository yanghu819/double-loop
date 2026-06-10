# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.2173, total=0.2173, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop1_loss=1.2231, loop_last_loss=0.2173, sec=1671.1
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5126
  future_seed: fs_gate=0.555, fs_update=1.000, fs_state_norm=8.888, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7524
  future_seed: fs_gate=0.555, fs_update=1.000, fs_state_norm=8.888, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 3: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.8364
  future_seed: fs_gate=0.555, fs_update=1.000, fs_state_norm=8.888, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 4: exact=0.0195, valid=0.0312, solved=0.0312, blank_acc=0.8462
  future_seed: fs_gate=0.555, fs_update=1.000, fs_state_norm=8.888, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 5: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479
  future_seed: fs_gate=0.555, fs_update=1.000, fs_state_norm=8.888, fs_decay=0.00, fb_in=0.000, fb_next=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0234, oracle_solved=0.0352, disagree=0.0000
  confidence: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479; gap=0.0000
  consistency: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479; gap=0.0000
  residual: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479; gap=0.0000
  majority: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479; gap=0.0000
- K4: oracle_exact=0.0234, oracle_solved=0.0352, disagree=0.0000
  confidence: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479; gap=0.0000
  consistency: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479; gap=0.0000
  residual: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479; gap=0.0000
  majority: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479; gap=0.0000
- K8: oracle_exact=0.0234, oracle_solved=0.0352, disagree=0.0000
  confidence: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479; gap=0.0000
  consistency: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479; gap=0.0000
  residual: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479; gap=0.0000
  majority: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K4: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K8: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0156, confidence=0.0156, residual=0.0156, disagree=0.0000
- K4: oracle_exact=0.0156, confidence=0.0156, residual=0.0156, disagree=0.0000
- K8: oracle_exact=0.0156, confidence=0.0156, residual=0.0156, disagree=0.0000
### loop5
- K1: oracle_exact=0.0234, confidence=0.0234, residual=0.0234, disagree=0.0000
- K4: oracle_exact=0.0234, confidence=0.0234, residual=0.0234, disagree=0.0000
- K8: oracle_exact=0.0234, confidence=0.0234, residual=0.0234, disagree=0.0000

## Case Bank

- h84: index=/huyang2/double-loop/.worktrees/h84shift-12x12-20260610T065757Z-0942939/runs/h84shift-probe-12x12-d256-l12-loop5-s1600-20260610T065757Z-0942939-cachebin/output/case_bank/h84/index.html; cases=/huyang2/double-loop/.worktrees/h84shift-12x12-20260610T065757Z-0942939/runs/h84shift-probe-12x12-d256-l12-loop5-s1600-20260610T065757Z-0942939-cachebin/output/case_bank/h84/cases.json; final_loop=5, exact=0.0352, blank_acc=0.8582, selected={'solved_by_loop': 6, 'almost_solved': 6, 'hard_failure': 6}

## Hole Transfer

- holes84: exact=0.0234, valid=0.0352, solved=0.0352, blank_acc=0.8479; indep=0.0000, exact/indep=24403.10
- holes60: exact=0.9297, valid=0.9570, solved=0.9570, blank_acc=0.9946; indep=0.7225, exact/indep=1.29
- holes72: exact=0.5469, valid=0.6094, solved=0.6094, blank_acc=0.9664; indep=0.0855, exact/indep=6.40
- holes96: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6202; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/h84shift-12x12-20260610T065757Z-0942939/runs/h84shift-probe-12x12-d256-l12-loop5-s1600-20260610T065757Z-0942939-cachebin/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/h84shift-12x12-20260610T065757Z-0942939/runs/h84shift-probe-12x12-d256-l12-loop5-s1600-20260610T065757Z-0942939-cachebin/output/case_bank
