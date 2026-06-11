# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.1178, total=0.1178, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, loop1_loss=1.8564, loop_last_loss=0.1178, sec=5130.8
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4001
  future_seed: fs_gate=0.599, fs_update=1.000, fs_state_norm=9.578, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6760
  future_seed: fs_gate=0.599, fs_update=1.000, fs_state_norm=9.578, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 3: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.8703
  future_seed: fs_gate=0.599, fs_update=1.000, fs_state_norm=9.578, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 4: exact=0.2715, valid=0.2715, solved=0.2715, blank_acc=0.9215
  future_seed: fs_gate=0.599, fs_update=1.000, fs_state_norm=9.578, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 5: exact=0.3164, valid=0.3184, solved=0.3184, blank_acc=0.9264
  future_seed: fs_gate=0.599, fs_update=1.000, fs_state_norm=9.578, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.3164, oracle_solved=0.3184, disagree=0.0000
  confidence: exact=0.3164, valid=0.3184, solved=0.3184, blank_acc=0.9264; gap=0.0000
  consistency: exact=0.3164, valid=0.3184, solved=0.3184, blank_acc=0.9264; gap=0.0000
  residual: exact=0.3164, valid=0.3184, solved=0.3184, blank_acc=0.9264; gap=0.0000
  majority: exact=0.3164, valid=0.3184, solved=0.3184, blank_acc=0.9264; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0293, confidence=0.0293, residual=0.0293, disagree=0.0000
### loop5
- K1: oracle_exact=0.3164, confidence=0.3164, residual=0.3164, disagree=0.0000

## Hole Transfer

- holes96: exact=0.3164, valid=0.3184, solved=0.3184, blank_acc=0.9264; indep=0.0007, exact/indep=486.46
- holes72: exact=0.9844, valid=0.9863, solved=0.9863, blank_acc=0.9993; indep=0.9523, exact/indep=1.03
- holes84: exact=0.8828, valid=0.8848, solved=0.8828, blank_acc=0.9927; indep=0.5425, exact/indep=1.63
- holes108: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5904; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/h96fullclean-20260611T1212Z-4337212/runs/h96fullclean-12x12-d256-l12-loop5-s4800-20260611T1212Z-4337212/output/futureseed_loop_case_seed52.html
