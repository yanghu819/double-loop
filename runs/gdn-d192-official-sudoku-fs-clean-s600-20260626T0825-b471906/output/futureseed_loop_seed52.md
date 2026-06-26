# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0479, total=1.0479, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.1842, loop_last_loss=1.0479, sec=407.3
- loop 1: exact=0.0034, valid=0.0034, solved=0.0034, blank_acc=0.4603, early=0.4439, late=0.4692, early_late_gap=-0.0253
  future_seed: fs_gate=0.491, fs_update=1.000, fs_state_norm=7.862, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0161, valid=0.0161, solved=0.0161, blank_acc=0.4995, early=0.4951, late=0.5053, early_late_gap=-0.0102
  future_seed: fs_gate=0.491, fs_update=1.000, fs_state_norm=7.862, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0176, valid=0.0176, solved=0.0176, blank_acc=0.5018, early=0.4970, late=0.5077, early_late_gap=-0.0108
  future_seed: fs_gate=0.491, fs_update=1.000, fs_state_norm=7.862, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0176, valid=0.0176, solved=0.0176, blank_acc=0.5018, early=0.4965, late=0.5081, early_late_gap=-0.0116
  future_seed: fs_gate=0.491, fs_update=1.000, fs_state_norm=7.862, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0176, valid=0.0176, solved=0.0176, blank_acc=0.5021, early=0.4969, late=0.5085, early_late_gap=-0.0116
  future_seed: fs_gate=0.491, fs_update=1.000, fs_state_norm=7.862, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-b471906-20260626/runs/gdn-d192-official-sudoku-fs-clean-s600-20260626T0825-b471906/output/futureseed_loop_case_seed52.html
