# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9991, total=0.9991, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.2564, loop_last_loss=0.9991, sec=2104.5
- loop 1: exact=0.0020, valid=0.0020, solved=0.0020, blank_acc=0.4481, early=0.4575, late=0.4454, early_late_gap=+0.0121
  future_seed: fs_gate=0.489, fs_update=1.000, fs_state_norm=7.828, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0288, valid=0.0288, solved=0.0288, blank_acc=0.5187, early=0.5158, late=0.5231, early_late_gap=-0.0073
  future_seed: fs_gate=0.489, fs_update=1.000, fs_state_norm=7.828, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0288, valid=0.0288, solved=0.0288, blank_acc=0.5288, early=0.5275, late=0.5320, early_late_gap=-0.0046
  future_seed: fs_gate=0.489, fs_update=1.000, fs_state_norm=7.828, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0288, valid=0.0288, solved=0.0288, blank_acc=0.5294, early=0.5283, late=0.5319, early_late_gap=-0.0036
  future_seed: fs_gate=0.489, fs_update=1.000, fs_state_norm=7.828, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0288, valid=0.0288, solved=0.0288, blank_acc=0.5300, early=0.5291, late=0.5329, early_late_gap=-0.0038
  future_seed: fs_gate=0.489, fs_update=1.000, fs_state_norm=7.828, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d256-fs-capacity-6ea0b0c-20260626/runs/gdn-d256-official-sudoku-fs-capacity-s1500-20260626T1515Z-6ea0b0c/output/futureseed_loop_case_seed52.html
