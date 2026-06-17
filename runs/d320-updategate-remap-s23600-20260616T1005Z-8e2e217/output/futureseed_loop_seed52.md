# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0884, total=0.2177, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=learned_gate, upd_h=0.983, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=0.7615, loop_last_loss=0.0884, sec=2565.0
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4854
  future_seed: fs_gate=0.621, fs_update=1.000, fs_state_norm=19.875, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.894, upd_h=0.788, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0254, valid=0.0254, solved=0.0254, blank_acc=0.7504
  future_seed: fs_gate=0.621, fs_update=1.000, fs_state_norm=19.875, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.779, upd_h=0.708, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.2734, valid=0.2754, solved=0.2754, blank_acc=0.8298
  future_seed: fs_gate=0.621, fs_update=1.000, fs_state_norm=19.875, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.811, upd_h=0.676, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.3145, valid=0.3262, solved=0.3262, blank_acc=0.8473
  future_seed: fs_gate=0.621, fs_update=1.000, fs_state_norm=19.875, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.975, upd_h=0.961, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.3281, valid=0.3398, solved=0.3398, blank_acc=0.8495
  future_seed: fs_gate=0.621, fs_update=1.000, fs_state_norm=19.875, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.984, upd_h=0.982, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.3340, valid=0.3457, solved=0.3457, blank_acc=0.8500
  future_seed: fs_gate=0.621, fs_update=1.000, fs_state_norm=19.875, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.984, upd_h=0.983, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.3340, oracle_solved=0.3457, disagree=0.0000
  confidence: exact=0.3340, valid=0.3457, solved=0.3457, blank_acc=0.8500; gap=0.0000
  consistency: exact=0.3340, valid=0.3457, solved=0.3457, blank_acc=0.8500; gap=0.0000
  residual: exact=0.3340, valid=0.3457, solved=0.3457, blank_acc=0.8500; gap=0.0000
  majority: exact=0.3340, valid=0.3457, solved=0.3457, blank_acc=0.8500; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.2734, confidence=0.2734, residual=0.2734, disagree=0.0000
### loop4
- K1: oracle_exact=0.3145, confidence=0.3145, residual=0.3145, disagree=0.0000
### loop5
- K1: oracle_exact=0.3281, confidence=0.3281, residual=0.3281, disagree=0.0000
### loop6
- K1: oracle_exact=0.3340, confidence=0.3340, residual=0.3340, disagree=0.0000

## Hole Transfer

- holes120: exact=0.3340, valid=0.3457, solved=0.3457, blank_acc=0.8500; indep=0.0000, exact/indep=98052413.68
- holes96: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9996; indep=0.9598, exact/indep=1.04
- holes108: exact=0.9551, valid=0.9551, solved=0.9551, blank_acc=0.9949; indep=0.5746, exact/indep=1.66
- holes132: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1696; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/d320-updategate-remap-s23600-20260616T1005Z-8e2e217/runs/d320-updategate-remap-s23600-20260616T1005Z-8e2e217/output/futureseed_loop_case_seed52.html
