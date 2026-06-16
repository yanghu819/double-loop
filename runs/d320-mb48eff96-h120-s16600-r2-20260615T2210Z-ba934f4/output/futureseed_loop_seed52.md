# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.1817, total=0.3530, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.0466, loop_last_loss=0.1817, sec=6784.7
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4108
  future_seed: fs_gate=0.617, fs_update=1.000, fs_state_norm=19.729, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0020, valid=0.0020, solved=0.0020, blank_acc=0.6655
  future_seed: fs_gate=0.617, fs_update=1.000, fs_state_norm=19.729, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1621, valid=0.1699, solved=0.1699, blank_acc=0.7732
  future_seed: fs_gate=0.617, fs_update=1.000, fs_state_norm=19.729, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.2188, valid=0.2324, solved=0.2324, blank_acc=0.7923
  future_seed: fs_gate=0.617, fs_update=1.000, fs_state_norm=19.729, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.2363, valid=0.2520, solved=0.2520, blank_acc=0.7955
  future_seed: fs_gate=0.617, fs_update=1.000, fs_state_norm=19.729, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.2363, valid=0.2520, solved=0.2520, blank_acc=0.7964
  future_seed: fs_gate=0.617, fs_update=1.000, fs_state_norm=19.729, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.2363, oracle_solved=0.2520, disagree=0.0000
  confidence: exact=0.2363, valid=0.2520, solved=0.2520, blank_acc=0.7964; gap=0.0000
  consistency: exact=0.2363, valid=0.2520, solved=0.2520, blank_acc=0.7964; gap=0.0000
  residual: exact=0.2363, valid=0.2520, solved=0.2520, blank_acc=0.7964; gap=0.0000
  majority: exact=0.2363, valid=0.2520, solved=0.2520, blank_acc=0.7964; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.1621, confidence=0.1621, residual=0.1621, disagree=0.0000
### loop4
- K1: oracle_exact=0.2188, confidence=0.2188, residual=0.2188, disagree=0.0000
### loop5
- K1: oracle_exact=0.2363, confidence=0.2363, residual=0.2363, disagree=0.0000
### loop6
- K1: oracle_exact=0.2363, confidence=0.2363, residual=0.2363, disagree=0.0000

## Hole Transfer

- holes120: exact=0.2363, valid=0.2520, solved=0.2520, blank_acc=0.7964; indep=0.0000, exact/indep=171964950602.69
- holes96: exact=0.9941, valid=0.9941, solved=0.9941, blank_acc=0.9993; indep=0.9357, exact/indep=1.06
- holes108: exact=0.9297, valid=0.9297, solved=0.9297, blank_acc=0.9940; indep=0.5239, exact/indep=1.77
- holes132: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1646; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/d320-mb48eff96-h120-s16600-r2-20260615T2210Z-ba934f4/runs/d320-mb48eff96-h120-s16600-r2-20260615T2210Z-ba934f4/output/futureseed_loop_case_seed52.html
