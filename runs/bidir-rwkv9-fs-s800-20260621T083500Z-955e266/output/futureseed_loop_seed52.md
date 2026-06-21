# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0097, total=0.0097, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=0.0305, loop_last_loss=0.0097, sec=459.1
- loop 1: exact=0.8047, valid=0.8047, solved=0.8047, blank_acc=0.9801, early=0.9798, late=0.9841, early_late_gap=-0.0043
  future_seed: fs_gate=0.523, fs_update=1.000, fs_state_norm=6.279, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.9375, valid=0.9375, solved=0.9375, blank_acc=0.9930, early=0.9933, late=0.9950, early_late_gap=-0.0018
  future_seed: fs_gate=0.523, fs_update=1.000, fs_state_norm=6.279, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.9512, valid=0.9512, solved=0.9512, blank_acc=0.9940, early=0.9947, late=0.9950, early_late_gap=-0.0003
  future_seed: fs_gate=0.523, fs_update=1.000, fs_state_norm=6.279, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.9492, valid=0.9492, solved=0.9492, blank_acc=0.9940, early=0.9947, late=0.9955, early_late_gap=-0.0008
  future_seed: fs_gate=0.523, fs_update=1.000, fs_state_norm=6.279, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.9492, valid=0.9492, solved=0.9492, blank_acc=0.9940, early=0.9947, late=0.9955, early_late_gap=-0.0008
  future_seed: fs_gate=0.523, fs_update=1.000, fs_state_norm=6.279, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.9492, oracle_solved=0.9492, disagree=0.0000
  confidence: exact=0.9492, valid=0.9492, solved=0.9492, blank_acc=0.9940, early=0.9947, late=0.9955, early_late_gap=-0.0008; gap=0.0000
  consistency: exact=0.9492, valid=0.9492, solved=0.9492, blank_acc=0.9940, early=0.9947, late=0.9955, early_late_gap=-0.0008; gap=0.0000
  residual: exact=0.9492, valid=0.9492, solved=0.9492, blank_acc=0.9940, early=0.9947, late=0.9955, early_late_gap=-0.0008; gap=0.0000
  majority: exact=0.9492, valid=0.9492, solved=0.9492, blank_acc=0.9940, early=0.9947, late=0.9955, early_late_gap=-0.0008; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.8047, confidence=0.8047, residual=0.8047, disagree=0.0000
### loop3
- K1: oracle_exact=0.9512, confidence=0.9512, residual=0.9512, disagree=0.0000
### loop5
- K1: oracle_exact=0.9492, confidence=0.9492, residual=0.9492, disagree=0.0000

## Hole Transfer

- holes12: exact=0.9492, valid=0.9492, solved=0.9492, blank_acc=0.9940, early=0.9947, late=0.9955, early_late_gap=-0.0008; indep=0.9301, exact/indep=1.02
- holes8: exact=0.9902, valid=0.9902, solved=0.9902, blank_acc=0.9983, early=0.9971, late=1.0000, early_late_gap=-0.0029; indep=0.9864, exact/indep=1.00
- holes16: exact=0.8008, valid=0.8008, solved=0.8008, blank_acc=0.9806, early=0.9799, late=0.9836, early_late_gap=-0.0037; indep=0.7308, exact/indep=1.10

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/futureseed-bidir-955e266-20260621T0850Z/runs/bidir-rwkv9-fs-s800-20260621T083500Z-955e266/output/futureseed_loop_case_seed52.html
