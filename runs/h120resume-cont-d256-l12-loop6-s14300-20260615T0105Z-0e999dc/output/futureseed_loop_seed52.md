# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0975, total=0.2439, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=0.8558, loop_last_loss=0.0975, sec=5239.8
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4147
  future_seed: fs_gate=0.685, fs_update=1.000, fs_state_norm=21.908, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0059, valid=0.0059, solved=0.0059, blank_acc=0.6417
  future_seed: fs_gate=0.685, fs_update=1.000, fs_state_norm=21.908, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1172, valid=0.1211, solved=0.1211, blank_acc=0.7309
  future_seed: fs_gate=0.685, fs_update=1.000, fs_state_norm=21.908, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.1621, valid=0.1719, solved=0.1719, blank_acc=0.7437
  future_seed: fs_gate=0.685, fs_update=1.000, fs_state_norm=21.908, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.1738, valid=0.1855, solved=0.1855, blank_acc=0.7460
  future_seed: fs_gate=0.685, fs_update=1.000, fs_state_norm=21.908, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.1777, valid=0.1895, solved=0.1895, blank_acc=0.7465
  future_seed: fs_gate=0.685, fs_update=1.000, fs_state_norm=21.908, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.1777, oracle_solved=0.1895, disagree=0.0000
  confidence: exact=0.1777, valid=0.1895, solved=0.1895, blank_acc=0.7465; gap=0.0000
  consistency: exact=0.1777, valid=0.1895, solved=0.1895, blank_acc=0.7465; gap=0.0000
  residual: exact=0.1777, valid=0.1895, solved=0.1895, blank_acc=0.7465; gap=0.0000
  majority: exact=0.1777, valid=0.1895, solved=0.1895, blank_acc=0.7465; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.1172, confidence=0.1172, residual=0.1172, disagree=0.0000
### loop6
- K1: oracle_exact=0.1777, confidence=0.1777, residual=0.1777, disagree=0.0000

## Hole Transfer

- holes120: exact=0.1777, valid=0.1895, solved=0.1895, blank_acc=0.7465; indep=0.0000, exact/indep=307906739017373.25
- holes96: exact=0.9980, valid=0.9980, solved=0.9980, blank_acc=1.0000; indep=1.0000, exact/indep=1.00
- holes108: exact=0.9473, valid=0.9492, solved=0.9492, blank_acc=0.9943; indep=0.5385, exact/indep=1.76
- holes132: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1560; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/h120resume-cont-d256-l12-loop6-s14300-20260615T0105Z-0e999dc/runs/h120resume-cont-d256-l12-loop6-s14300-20260615T0105Z-0e999dc/output/futureseed_loop_case_seed52.html
