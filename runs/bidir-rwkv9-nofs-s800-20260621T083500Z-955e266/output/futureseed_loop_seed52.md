# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.4779, total=0.4779, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=0.4806, loop_last_loss=0.4779, sec=467.0
- loop 1: exact=0.0098, valid=0.0098, solved=0.0098, blank_acc=0.6649, early=0.4596, late=0.8347, early_late_gap=-0.3752
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0137, valid=0.0137, solved=0.0137, blank_acc=0.6670, early=0.4649, late=0.8352, early_late_gap=-0.3704
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0137, valid=0.0137, solved=0.0137, blank_acc=0.6678, early=0.4682, late=0.8377, early_late_gap=-0.3695
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6686, early=0.4692, late=0.8377, early_late_gap=-0.3685
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6688, early=0.4697, late=0.8382, early_late_gap=-0.3685
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0156, oracle_solved=0.0156, disagree=0.0000
  confidence: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6688, early=0.4697, late=0.8382, early_late_gap=-0.3685; gap=0.0000
  consistency: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6688, early=0.4697, late=0.8382, early_late_gap=-0.3685; gap=0.0000
  residual: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6688, early=0.4697, late=0.8382, early_late_gap=-0.3685; gap=0.0000
  majority: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6688, early=0.4697, late=0.8382, early_late_gap=-0.3685; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0098, confidence=0.0098, residual=0.0098, disagree=0.0000
### loop3
- K1: oracle_exact=0.0137, confidence=0.0137, residual=0.0137, disagree=0.0000
### loop5
- K1: oracle_exact=0.0156, confidence=0.0156, residual=0.0156, disagree=0.0000

## Hole Transfer

- holes12: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6688, early=0.4697, late=0.8382, early_late_gap=-0.3685; indep=0.0080, exact/indep=1.95
- holes8: exact=0.0625, valid=0.0625, solved=0.0625, blank_acc=0.6934, early=0.4779, late=0.8583, early_late_gap=-0.3804; indep=0.0534, exact/indep=1.17
- holes16: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6360, early=0.4155, late=0.8199, early_late_gap=-0.4044; indep=0.0007, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/futureseed-bidir-955e266-20260621T0850Z/runs/bidir-rwkv9-nofs-s800-20260621T083500Z-955e266/output/futureseed_loop_case_seed52.html
