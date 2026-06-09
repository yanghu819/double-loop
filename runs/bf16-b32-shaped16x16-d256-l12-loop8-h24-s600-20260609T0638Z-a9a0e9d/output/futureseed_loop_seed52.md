# 16x16 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 16x16, box: 4x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9751, total=0.9753, loop_loss=shaped, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop1_loss=0.9817, loop_last_loss=0.9751, sec=969.6
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3678
  future_seed: fs_gate=0.497, fs_update=0.545, fs_state_norm=15.904, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3706
  future_seed: fs_gate=0.497, fs_update=0.545, fs_state_norm=15.904, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3714
  future_seed: fs_gate=0.497, fs_update=0.545, fs_state_norm=15.904, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3716
  future_seed: fs_gate=0.497, fs_update=0.545, fs_state_norm=15.904, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3716
  future_seed: fs_gate=0.497, fs_update=0.545, fs_state_norm=15.904, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 6: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3714
  future_seed: fs_gate=0.497, fs_update=0.545, fs_state_norm=15.904, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 7: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3713
  future_seed: fs_gate=0.497, fs_update=0.545, fs_state_norm=15.904, fs_decay=0.50, fb_in=0.000, fb_next=0.000
- loop 8: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3711
  future_seed: fs_gate=0.497, fs_update=0.545, fs_state_norm=15.904, fs_decay=0.50, fb_in=0.000, fb_next=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3711; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3711; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3711; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3711; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop5
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop8
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000

## Hole Transfer

- holes24: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3711; indep=0.0000, exact/indep=0.00
- holes16: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4043; indep=0.0000, exact/indep=0.00
- holes32: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3571; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/bf16-scale-20260609T0610Z-a9a0e9d/runs/bf16-b32-shaped16x16-d256-l12-loop8-h24-s600-20260609T0638Z-a9a0e9d/output/futureseed_loop_case_seed52.html
