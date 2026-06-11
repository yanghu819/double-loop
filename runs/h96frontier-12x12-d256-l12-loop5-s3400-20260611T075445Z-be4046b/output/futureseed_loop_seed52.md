# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.3931, total=0.3931, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop1_loss=1.6003, loop_last_loss=0.3931, sec=3856.5
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4211
  future_seed: fs_gate=0.582, fs_update=1.000, fs_state_norm=9.316, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6319
  future_seed: fs_gate=0.582, fs_update=1.000, fs_state_norm=9.316, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7414
  future_seed: fs_gate=0.582, fs_update=1.000, fs_state_norm=9.316, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 4: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.7732
  future_seed: fs_gate=0.582, fs_update=1.000, fs_state_norm=9.316, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 5: exact=0.0104, valid=0.0104, solved=0.0104, blank_acc=0.7769
  future_seed: fs_gate=0.582, fs_update=1.000, fs_state_norm=9.316, fs_decay=0.00, fb_in=0.000, fb_next=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0104, oracle_solved=0.0104, disagree=0.0000
  confidence: exact=0.0104, valid=0.0104, solved=0.0104, blank_acc=0.7769; gap=0.0000
  consistency: exact=0.0104, valid=0.0104, solved=0.0104, blank_acc=0.7769; gap=0.0000
  residual: exact=0.0104, valid=0.0104, solved=0.0104, blank_acc=0.7769; gap=0.0000
  majority: exact=0.0104, valid=0.0104, solved=0.0104, blank_acc=0.7769; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop5
- K1: oracle_exact=0.0104, confidence=0.0104, residual=0.0104, disagree=0.0000

## Case Bank

- h96: index=/huyang2/double-loop/.worktrees/h96frontier-12x12-20260611T075445Z-be4046b/runs/h96frontier-12x12-d256-l12-loop5-s3400-20260611T075445Z-be4046b/output/case_bank/h96/index.html; cases=/huyang2/double-loop/.worktrees/h96frontier-12x12-20260611T075445Z-be4046b/runs/h96frontier-12x12-d256-l12-loop5-s3400-20260611T075445Z-be4046b/output/case_bank/h96/cases.json; final_loop=5, exact=0.0068, blank_acc=0.7652, selected={'solved_by_loop': 7, 'almost_solved': 8, 'hard_failure': 8}

## Hole Transfer

- holes96: exact=0.0104, valid=0.0104, solved=0.0104, blank_acc=0.7769; indep=0.0000, exact/indep=350805953.39
- holes72: exact=0.8750, valid=0.9323, solved=0.9323, blank_acc=0.9889; indep=0.4487, exact/indep=1.95
- holes84: exact=0.4740, valid=0.5469, solved=0.5469, blank_acc=0.9515; indep=0.0154, exact/indep=30.83
- holes108: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4829; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/h96frontier-12x12-20260611T075445Z-be4046b/runs/h96frontier-12x12-d256-l12-loop5-s3400-20260611T075445Z-be4046b/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/h96frontier-12x12-20260611T075445Z-be4046b/runs/h96frontier-12x12-d256-l12-loop5-s3400-20260611T075445Z-be4046b/output/case_bank
