# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.3605, total=0.5267, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, loop1_loss=1.1588, loop_last_loss=0.3605, sec=8556.3
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3441
  future_seed: fs_gate=0.650, fs_update=1.000, fs_state_norm=20.786, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4916
  future_seed: fs_gate=0.650, fs_update=1.000, fs_state_norm=20.786, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 3: exact=0.0117, valid=0.0137, solved=0.0137, blank_acc=0.5719
  future_seed: fs_gate=0.650, fs_update=1.000, fs_state_norm=20.786, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 4: exact=0.0273, valid=0.0293, solved=0.0293, blank_acc=0.5843
  future_seed: fs_gate=0.650, fs_update=1.000, fs_state_norm=20.786, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 5: exact=0.0273, valid=0.0312, solved=0.0312, blank_acc=0.5860
  future_seed: fs_gate=0.650, fs_update=1.000, fs_state_norm=20.786, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 6: exact=0.0273, valid=0.0312, solved=0.0312, blank_acc=0.5866
  future_seed: fs_gate=0.650, fs_update=1.000, fs_state_norm=20.786, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0273, oracle_solved=0.0312, disagree=0.0000
  confidence: exact=0.0273, valid=0.0312, solved=0.0312, blank_acc=0.5866; gap=0.0000
  consistency: exact=0.0273, valid=0.0312, solved=0.0312, blank_acc=0.5866; gap=0.0000
  residual: exact=0.0273, valid=0.0312, solved=0.0312, blank_acc=0.5866; gap=0.0000
  majority: exact=0.0273, valid=0.0312, solved=0.0312, blank_acc=0.5866; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0117, confidence=0.0117, residual=0.0117, disagree=0.0000
### loop5
- K1: oracle_exact=0.0273, confidence=0.0273, residual=0.0273, disagree=0.0000
### loop6
- K1: oracle_exact=0.0273, confidence=0.0273, residual=0.0273, disagree=0.0000

## Hole Transfer

- holes120: exact=0.0273, valid=0.0312, solved=0.0312, blank_acc=0.5866; indep=0.0000, exact/indep=171370466415817443407560704.00
- holes96: exact=0.9922, valid=0.9922, solved=0.9922, blank_acc=0.9999; indep=0.9903, exact/indep=1.00
- holes108: exact=0.9160, valid=0.9277, solved=0.9277, blank_acc=0.9924; indep=0.4406, exact/indep=2.08
- holes132: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1530; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/h120loopall-retry-20260611T1640Z-8ccc4a9/runs/h120loopall-retry-12x12-d256-l12-loop6-s7400-20260611T1640Z-8ccc4a9/output/futureseed_loop_case_seed52.html
