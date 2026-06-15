# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.2807, total=0.5065, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.3199, loop_last_loss=0.2807, sec=9876.7
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3384
  future_seed: fs_gate=0.598, fs_update=1.000, fs_state_norm=19.126, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5104
  future_seed: fs_gate=0.598, fs_update=1.000, fs_state_norm=19.126, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0098, valid=0.0137, solved=0.0137, blank_acc=0.6181
  future_seed: fs_gate=0.598, fs_update=1.000, fs_state_norm=19.126, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0488, valid=0.0605, solved=0.0605, blank_acc=0.6435
  future_seed: fs_gate=0.598, fs_update=1.000, fs_state_norm=19.126, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0547, valid=0.0664, solved=0.0664, blank_acc=0.6473
  future_seed: fs_gate=0.598, fs_update=1.000, fs_state_norm=19.126, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.0547, valid=0.0664, solved=0.0664, blank_acc=0.6484
  future_seed: fs_gate=0.598, fs_update=1.000, fs_state_norm=19.126, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0547, oracle_solved=0.0664, disagree=0.0000
  confidence: exact=0.0547, valid=0.0664, solved=0.0664, blank_acc=0.6484; gap=0.0000
  consistency: exact=0.0547, valid=0.0664, solved=0.0664, blank_acc=0.6484; gap=0.0000
  residual: exact=0.0547, valid=0.0664, solved=0.0664, blank_acc=0.6484; gap=0.0000
  majority: exact=0.0547, valid=0.0664, solved=0.0664, blank_acc=0.6484; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0098, confidence=0.0098, residual=0.0098, disagree=0.0000
### loop6
- K1: oracle_exact=0.0547, confidence=0.0547, residual=0.0547, disagree=0.0000

## Hole Transfer

- holes120: exact=0.0547, valid=0.0664, solved=0.0664, blank_acc=0.6484; indep=0.0000, exact/indep=2071710187714830925824.00
- holes96: exact=0.9785, valid=0.9785, solved=0.9785, blank_acc=0.9989; indep=0.9034, exact/indep=1.08
- holes108: exact=0.9004, valid=0.9082, solved=0.9082, blank_acc=0.9884; indep=0.2839, exact/indep=3.17
- holes132: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1659; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/d320-effb72-h120-s9800-20260615T1230Z-785f3cd/runs/d320-effb72-h120-resume7000-s9800-20260615T1358Z-785f3cd/output/futureseed_loop_case_seed52.html
