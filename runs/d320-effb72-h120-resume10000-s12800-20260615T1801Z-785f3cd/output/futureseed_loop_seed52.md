# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.2818, total=0.4604, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.1807, loop_last_loss=0.2818, sec=9339.6
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3711
  future_seed: fs_gate=0.606, fs_update=1.000, fs_state_norm=19.398, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0020, valid=0.0020, solved=0.0020, blank_acc=0.5832
  future_seed: fs_gate=0.606, fs_update=1.000, fs_state_norm=19.398, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0664, valid=0.0684, solved=0.0684, blank_acc=0.7041
  future_seed: fs_gate=0.606, fs_update=1.000, fs_state_norm=19.398, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.1074, valid=0.1113, solved=0.1113, blank_acc=0.7289
  future_seed: fs_gate=0.606, fs_update=1.000, fs_state_norm=19.398, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.1191, valid=0.1230, solved=0.1230, blank_acc=0.7319
  future_seed: fs_gate=0.606, fs_update=1.000, fs_state_norm=19.398, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.1270, valid=0.1309, solved=0.1309, blank_acc=0.7331
  future_seed: fs_gate=0.606, fs_update=1.000, fs_state_norm=19.398, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.1270, oracle_solved=0.1309, disagree=0.0000
  confidence: exact=0.1270, valid=0.1309, solved=0.1309, blank_acc=0.7331; gap=0.0000
  consistency: exact=0.1270, valid=0.1309, solved=0.1309, blank_acc=0.7331; gap=0.0000
  residual: exact=0.1270, valid=0.1309, solved=0.1309, blank_acc=0.7331; gap=0.0000
  majority: exact=0.1270, valid=0.1309, solved=0.1309, blank_acc=0.7331; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0664, confidence=0.0664, residual=0.0664, disagree=0.0000
### loop4
- K1: oracle_exact=0.1074, confidence=0.1074, residual=0.1074, disagree=0.0000
### loop5
- K1: oracle_exact=0.1191, confidence=0.1191, residual=0.1191, disagree=0.0000
### loop6
- K1: oracle_exact=0.1270, confidence=0.1270, residual=0.1270, disagree=0.0000

## Hole Transfer

- holes120: exact=0.1270, valid=0.1309, solved=0.1309, blank_acc=0.7331; indep=0.0000, exact/indep=1916508270101759.50
- holes96: exact=0.9883, valid=0.9883, solved=0.9883, blank_acc=0.9993; indep=0.9376, exact/indep=1.05
- holes108: exact=0.9355, valid=0.9375, solved=0.9375, blank_acc=0.9938; indep=0.5127, exact/indep=1.82
- holes132: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1642; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/d320-effb72-h120-s12800-20260615T1703Z-785f3cd/runs/d320-effb72-h120-resume10000-s12800-20260615T1801Z-785f3cd/output/futureseed_loop_case_seed52.html
