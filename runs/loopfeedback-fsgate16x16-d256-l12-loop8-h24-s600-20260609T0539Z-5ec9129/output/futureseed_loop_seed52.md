# 16x16 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 16x16, box: 4x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9810, total=0.9809, loop_loss=shaped, noise=feature_diff, buffer=8192, fs_update=learned, loop_fb=1.00, loop1_loss=0.9815, loop_last_loss=0.9810, sec=973.0
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3249
  future_seed: fs_gate=0.498, fs_update=0.543, fs_state_norm=15.942, fs_decay=0.50, fb_in=0.000, fb_next=1.537
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3270
  future_seed: fs_gate=0.498, fs_update=0.543, fs_state_norm=15.942, fs_decay=0.50, fb_in=1.537, fb_next=1.540
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3265
  future_seed: fs_gate=0.498, fs_update=0.543, fs_state_norm=15.942, fs_decay=0.50, fb_in=1.540, fb_next=1.541
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3265
  future_seed: fs_gate=0.498, fs_update=0.543, fs_state_norm=15.942, fs_decay=0.50, fb_in=1.541, fb_next=1.541
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3267
  future_seed: fs_gate=0.498, fs_update=0.543, fs_state_norm=15.942, fs_decay=0.50, fb_in=1.541, fb_next=1.541
- loop 6: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3268
  future_seed: fs_gate=0.498, fs_update=0.543, fs_state_norm=15.942, fs_decay=0.50, fb_in=1.541, fb_next=1.541
- loop 7: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3270
  future_seed: fs_gate=0.498, fs_update=0.543, fs_state_norm=15.942, fs_decay=0.50, fb_in=1.541, fb_next=1.541
- loop 8: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3270
  future_seed: fs_gate=0.498, fs_update=0.543, fs_state_norm=15.942, fs_decay=0.50, fb_in=1.541, fb_next=1.541

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3270; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3270; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3270; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3270; gap=0.0000

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

- holes24: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3270; indep=0.0000, exact/indep=0.00
- holes16: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3501; indep=0.0000, exact/indep=0.00
- holes32: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2990; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/loopfeedback-fsgate-20260609T0534Z-5ec9129/runs/loopfeedback-fsgate16x16-d256-l12-loop8-h24-s600-20260609T0539Z-5ec9129/output/futureseed_loop_case_seed52.html
