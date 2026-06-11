# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.3850, total=0.3850, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, loop1_loss=1.4437, loop_last_loss=0.3850, sec=2556.2
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4309
  future_seed: fs_gate=0.575, fs_update=1.000, fs_state_norm=9.198, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6253
  future_seed: fs_gate=0.575, fs_update=1.000, fs_state_norm=9.198, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7348
  future_seed: fs_gate=0.575, fs_update=1.000, fs_state_norm=9.198, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7522
  future_seed: fs_gate=0.575, fs_update=1.000, fs_state_norm=9.198, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7545
  future_seed: fs_gate=0.575, fs_update=1.000, fs_state_norm=9.198, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7545; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7545; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7545; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7545; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop5
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000

## Hole Transfer

- holes96: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7545; indep=0.0000, exact/indep=0.00
- holes72: exact=0.9010, valid=0.9323, solved=0.9323, blank_acc=0.9915; indep=0.5409, exact/indep=1.67
- holes84: exact=0.3828, valid=0.4036, solved=0.4036, blank_acc=0.9445; indep=0.0082, exact/indep=46.45
- holes108: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4597; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/h96focus-20260611T1046Z-edb4804/runs/h96focus-12x12-d256-l12-loop5-s2400-20260611T1046Z-edb4804/output/futureseed_loop_case_seed52.html
