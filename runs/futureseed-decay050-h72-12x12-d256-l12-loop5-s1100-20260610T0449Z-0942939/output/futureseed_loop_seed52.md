# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.1566, total=0.1566, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop1_loss=0.6853, loop_last_loss=0.1566, sec=1213.6
- loop 1: exact=0.0026, valid=0.0026, solved=0.0026, blank_acc=0.8402
  future_seed: fs_gate=0.519, fs_update=0.545, fs_state_norm=8.300, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 2: exact=0.3464, valid=0.3516, solved=0.3516, blank_acc=0.9657
  future_seed: fs_gate=0.519, fs_update=0.545, fs_state_norm=8.300, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 3: exact=0.4896, valid=0.5156, solved=0.5156, blank_acc=0.9726
  future_seed: fs_gate=0.519, fs_update=0.545, fs_state_norm=8.300, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 4: exact=0.5078, valid=0.5365, solved=0.5365, blank_acc=0.9734
  future_seed: fs_gate=0.519, fs_update=0.545, fs_state_norm=8.300, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 5: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739
  future_seed: fs_gate=0.519, fs_update=0.545, fs_state_norm=8.300, fs_decay=0.50, fb_in=0.000, fb_next=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.5052, oracle_solved=0.5365, disagree=0.0000
  confidence: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739; gap=0.0000
  consistency: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739; gap=0.0000
  residual: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739; gap=0.0000
  majority: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739; gap=0.0000
- K4: oracle_exact=0.5052, oracle_solved=0.5365, disagree=0.0000
  confidence: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739; gap=0.0000
  consistency: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739; gap=0.0000
  residual: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739; gap=0.0000
  majority: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739; gap=0.0000
- K8: oracle_exact=0.5052, oracle_solved=0.5365, disagree=0.0000
  confidence: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739; gap=0.0000
  consistency: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739; gap=0.0000
  residual: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739; gap=0.0000
  majority: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0026, confidence=0.0026, residual=0.0026, disagree=0.0000
- K4: oracle_exact=0.0026, confidence=0.0026, residual=0.0026, disagree=0.0000
- K8: oracle_exact=0.0026, confidence=0.0026, residual=0.0026, disagree=0.0000
### loop3
- K1: oracle_exact=0.4896, confidence=0.4896, residual=0.4896, disagree=0.0000
- K4: oracle_exact=0.4896, confidence=0.4896, residual=0.4896, disagree=0.0000
- K8: oracle_exact=0.4896, confidence=0.4896, residual=0.4896, disagree=0.0000
### loop5
- K1: oracle_exact=0.5052, confidence=0.5052, residual=0.5052, disagree=0.0000
- K4: oracle_exact=0.5052, confidence=0.5052, residual=0.5052, disagree=0.0000
- K8: oracle_exact=0.5052, confidence=0.5052, residual=0.5052, disagree=0.0000

## Case Bank

- h72: index=/huyang2/double-loop/.worktrees/fsdecay-h72-20260610T044909Z-0942939/runs/futureseed-decay050-h72-12x12-d256-l12-loop5-s1100-20260610T0449Z-0942939/output/case_bank/h72/index.html; cases=/huyang2/double-loop/.worktrees/fsdecay-h72-20260610T044909Z-0942939/runs/futureseed-decay050-h72-12x12-d256-l12-loop5-s1100-20260610T0449Z-0942939/output/case_bank/h72/cases.json; final_loop=5, exact=0.0635, blank_acc=0.9007, selected={'solved_by_loop': 8, 'almost_solved': 8, 'hard_failure': 8}

## Hole Transfer

- holes60: exact=0.5052, valid=0.5365, solved=0.5365, blank_acc=0.9739; indep=0.2042, exact/indep=2.47
- holes48: exact=0.9062, valid=0.9193, solved=0.9193, blank_acc=0.9934; indep=0.7290, exact/indep=1.24
- holes72: exact=0.0521, valid=0.0651, solved=0.0651, blank_acc=0.9007; indep=0.0005, exact/indep=97.19
- holes84: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7255; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/fsdecay-h72-20260610T044909Z-0942939/runs/futureseed-decay050-h72-12x12-d256-l12-loop5-s1100-20260610T0449Z-0942939/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/fsdecay-h72-20260610T044909Z-0942939/runs/futureseed-decay050-h72-12x12-d256-l12-loop5-s1100-20260610T0449Z-0942939/output/case_bank
