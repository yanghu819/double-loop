# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.2304, total=0.2304, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=learned, loop_fb=0.00, loop1_loss=0.8977, loop_last_loss=0.2304, sec=1171.1
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7696
  future_seed: fs_gate=0.525, fs_update=0.547, fs_state_norm=8.405, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 2: exact=0.0781, valid=0.0781, solved=0.0781, blank_acc=0.9301
  future_seed: fs_gate=0.525, fs_update=0.547, fs_state_norm=8.405, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 3: exact=0.1562, valid=0.1589, solved=0.1589, blank_acc=0.9471
  future_seed: fs_gate=0.525, fs_update=0.547, fs_state_norm=8.405, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 4: exact=0.1771, valid=0.1797, solved=0.1797, blank_acc=0.9493
  future_seed: fs_gate=0.525, fs_update=0.547, fs_state_norm=8.405, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 5: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502
  future_seed: fs_gate=0.525, fs_update=0.547, fs_state_norm=8.405, fs_decay=0.50, fb_in=0.000, fb_next=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.1927, oracle_solved=0.1953, disagree=0.0000
  confidence: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502; gap=0.0000
  consistency: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502; gap=0.0000
  residual: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502; gap=0.0000
  majority: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502; gap=0.0000
- K4: oracle_exact=0.1927, oracle_solved=0.1953, disagree=0.0000
  confidence: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502; gap=0.0000
  consistency: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502; gap=0.0000
  residual: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502; gap=0.0000
  majority: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502; gap=0.0000
- K8: oracle_exact=0.1927, oracle_solved=0.1953, disagree=0.0000
  confidence: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502; gap=0.0000
  consistency: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502; gap=0.0000
  residual: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502; gap=0.0000
  majority: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K4: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
- K8: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.1562, confidence=0.1562, residual=0.1562, disagree=0.0000
- K4: oracle_exact=0.1562, confidence=0.1562, residual=0.1562, disagree=0.0000
- K8: oracle_exact=0.1562, confidence=0.1562, residual=0.1562, disagree=0.0000
### loop5
- K1: oracle_exact=0.1927, confidence=0.1927, residual=0.1927, disagree=0.0000
- K4: oracle_exact=0.1927, confidence=0.1927, residual=0.1927, disagree=0.0000
- K8: oracle_exact=0.1927, confidence=0.1927, residual=0.1927, disagree=0.0000

## Case Bank

- h72: index=/huyang2/double-loop/.worktrees/fslearned-h72-20260610T051555Z-0942939/runs/futureseed-learnedupdate-h72-12x12-d256-l12-loop5-s1100-20260610T0513Z-0942939/output/case_bank/h72/index.html; cases=/huyang2/double-loop/.worktrees/fslearned-h72-20260610T051555Z-0942939/runs/futureseed-learnedupdate-h72-12x12-d256-l12-loop5-s1100-20260610T0513Z-0942939/output/case_bank/h72/cases.json; final_loop=5, exact=0.0020, blank_acc=0.8553, selected={'solved_by_loop': 2, 'almost_solved': 8, 'hard_failure': 8}

## Hole Transfer

- holes60: exact=0.1927, valid=0.1953, solved=0.1953, blank_acc=0.9502; indep=0.0466, exact/indep=4.14
- holes48: exact=0.7005, valid=0.7083, solved=0.7083, blank_acc=0.9859; indep=0.5070, exact/indep=1.38
- holes72: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.8530; indep=0.0000, exact/indep=0.00
- holes84: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6809; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/fslearned-h72-20260610T051555Z-0942939/runs/futureseed-learnedupdate-h72-12x12-d256-l12-loop5-s1100-20260610T0513Z-0942939/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/fslearned-h72-20260610T051555Z-0942939/runs/futureseed-learnedupdate-h72-12x12-d256-l12-loop5-s1100-20260610T0513Z-0942939/output/case_bank
