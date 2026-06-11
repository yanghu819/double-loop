# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.1359, total=0.1359, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, loop1_loss=2.1065, loop_last_loss=0.1359, sec=6591.7
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3186
  future_seed: fs_gate=0.610, fs_update=1.000, fs_state_norm=9.752, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5494
  future_seed: fs_gate=0.610, fs_update=1.000, fs_state_norm=9.752, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 3: exact=0.0020, valid=0.0020, solved=0.0020, blank_acc=0.7876
  future_seed: fs_gate=0.610, fs_update=1.000, fs_state_norm=9.752, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 4: exact=0.1328, valid=0.1348, solved=0.1348, blank_acc=0.8588
  future_seed: fs_gate=0.610, fs_update=1.000, fs_state_norm=9.752, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 5: exact=0.1797, valid=0.1816, solved=0.1816, blank_acc=0.8685
  future_seed: fs_gate=0.610, fs_update=1.000, fs_state_norm=9.752, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.1797, oracle_solved=0.1816, disagree=0.0000
  confidence: exact=0.1797, valid=0.1816, solved=0.1816, blank_acc=0.8685; gap=0.0000
  consistency: exact=0.1797, valid=0.1816, solved=0.1816, blank_acc=0.8685; gap=0.0000
  residual: exact=0.1797, valid=0.1816, solved=0.1816, blank_acc=0.8685; gap=0.0000
  majority: exact=0.1797, valid=0.1816, solved=0.1816, blank_acc=0.8685; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0020, confidence=0.0020, residual=0.0020, disagree=0.0000
### loop5
- K1: oracle_exact=0.1797, confidence=0.1797, residual=0.1797, disagree=0.0000

## Hole Transfer

- holes108: exact=0.1797, valid=0.1816, solved=0.1816, blank_acc=0.8685; indep=0.0000, exact/indep=739462.26
- holes84: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9992; indep=0.9357, exact/indep=1.06
- holes96: exact=0.8848, valid=0.8906, solved=0.8906, blank_acc=0.9934; indep=0.5289, exact/indep=1.67
- holes120: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3439; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/h108frontier-20260611T1400Z-e446f4a/runs/h108frontier-12x12-d256-l12-loop5-s6200-20260611T1400Z-e446f4a/output/futureseed_loop_case_seed52.html
