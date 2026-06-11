# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.8789, total=0.8789, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop1_loss=1.2935, loop_last_loss=0.8789, sec=861.2
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4567
  future_seed: fs_gate=0.511, fs_update=1.000, fs_state_norm=8.175, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5440
  future_seed: fs_gate=0.511, fs_update=1.000, fs_state_norm=8.175, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5505
  future_seed: fs_gate=0.511, fs_update=1.000, fs_state_norm=8.175, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5519
  future_seed: fs_gate=0.511, fs_update=1.000, fs_state_norm=8.175, fs_decay=0.00, fb_in=0.000, fb_next=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5523
  future_seed: fs_gate=0.511, fs_update=1.000, fs_state_norm=8.175, fs_decay=0.00, fb_in=0.000, fb_next=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5523; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5523; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5523; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5523; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop5
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000

## Hole Transfer

- holes84: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5523; indep=0.0000, exact/indep=0.00
- holes72: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6675; indep=0.0000, exact/indep=0.00
- holes96: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4264; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/capprobe-d320-h84-20260611T054106Z-d29bb4e/runs/capprobe-d320-12x12-h84-s900-20260611T054106Z-d29bb4e/output/futureseed_loop_case_seed52.html
