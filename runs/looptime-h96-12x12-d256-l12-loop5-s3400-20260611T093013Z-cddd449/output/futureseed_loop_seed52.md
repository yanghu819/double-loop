# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.3667, total=0.3667, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=1.00, loop1_loss=1.2383, loop_last_loss=0.3667, sec=3651.6
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5102
  future_seed: fs_gate=0.567, fs_update=1.000, fs_state_norm=9.074, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=2.012
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6538
  future_seed: fs_gate=0.567, fs_update=1.000, fs_state_norm=9.074, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=1.729
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7373
  future_seed: fs_gate=0.567, fs_update=1.000, fs_state_norm=9.074, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=1.099
- loop 4: exact=0.0026, valid=0.0052, solved=0.0052, blank_acc=0.7527
  future_seed: fs_gate=0.567, fs_update=1.000, fs_state_norm=9.074, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=1.076
- loop 5: exact=0.0052, valid=0.0104, solved=0.0104, blank_acc=0.7549
  future_seed: fs_gate=0.567, fs_update=1.000, fs_state_norm=9.074, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=1.706

## Stochastic Rollouts

- K1: oracle_exact=0.0052, oracle_solved=0.0104, disagree=0.0000
  confidence: exact=0.0052, valid=0.0104, solved=0.0104, blank_acc=0.7549; gap=0.0000
  consistency: exact=0.0052, valid=0.0104, solved=0.0104, blank_acc=0.7549; gap=0.0000
  residual: exact=0.0052, valid=0.0104, solved=0.0104, blank_acc=0.7549; gap=0.0000
  majority: exact=0.0052, valid=0.0104, solved=0.0104, blank_acc=0.7549; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop5
- K1: oracle_exact=0.0052, confidence=0.0052, residual=0.0052, disagree=0.0000

## Case Bank

- h96: index=/huyang2/double-loop/.worktrees/looptime-h96-20260611T093013Z-cddd449/runs/looptime-h96-12x12-d256-l12-loop5-s3400-20260611T093013Z-cddd449/output/case_bank/h96/index.html; cases=/huyang2/double-loop/.worktrees/looptime-h96-20260611T093013Z-cddd449/runs/looptime-h96-12x12-d256-l12-loop5-s3400-20260611T093013Z-cddd449/output/case_bank/h96/cases.json; final_loop=5, exact=0.0059, blank_acc=0.7490, selected={'solved_by_loop': 6, 'almost_solved': 8, 'hard_failure': 8}

## Hole Transfer

- holes96: exact=0.0052, valid=0.0104, solved=0.0104, blank_acc=0.7549; indep=0.0000, exact/indep=2736954759.20
- holes72: exact=0.8542, valid=0.9453, solved=0.9453, blank_acc=0.9868; indep=0.3841, exact/indep=2.22
- holes84: exact=0.4115, valid=0.4714, solved=0.4714, blank_acc=0.9397; indep=0.0054, exact/indep=76.20
- holes108: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4698; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/looptime-h96-20260611T093013Z-cddd449/runs/looptime-h96-12x12-d256-l12-loop5-s3400-20260611T093013Z-cddd449/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/looptime-h96-20260611T093013Z-cddd449/runs/looptime-h96-12x12-d256-l12-loop5-s3400-20260611T093013Z-cddd449/output/case_bank
