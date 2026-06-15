# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.5860, total=0.7857, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.4665, loop_last_loss=0.5860, sec=9251.6
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3068
  future_seed: fs_gate=0.588, fs_update=1.000, fs_state_norm=18.822, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4235
  future_seed: fs_gate=0.588, fs_update=1.000, fs_state_norm=18.822, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4922
  future_seed: fs_gate=0.588, fs_update=1.000, fs_state_norm=18.822, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5065
  future_seed: fs_gate=0.588, fs_update=1.000, fs_state_norm=18.822, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5095
  future_seed: fs_gate=0.588, fs_update=1.000, fs_state_norm=18.822, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5102
  future_seed: fs_gate=0.588, fs_update=1.000, fs_state_norm=18.822, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5102; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5102; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5102; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5102; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop4
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop5
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop6
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000

## Hole Transfer

- holes120: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5102; indep=0.0000, exact/indep=0.00
- holes96: exact=0.9805, valid=0.9824, solved=0.9824, blank_acc=0.9982; indep=0.8387, exact/indep=1.17
- holes108: exact=0.7344, valid=0.7461, solved=0.7461, blank_acc=0.9694; indep=0.0348, exact/indep=21.11
- holes132: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1635; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/d320-effb72-h120-step6800-20260615T0502Z-785f3cd/runs/d320-effb72-h120-resume4000-s6800-20260615T0916Z-785f3cd/output/futureseed_loop_case_seed52.html
