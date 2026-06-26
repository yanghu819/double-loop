# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.6369, total=1.6369, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.6407, loop_last_loss=1.6369, sec=442.6
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2663, early=0.2045, late=0.3327, early_late_gap=-0.1282
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2677, early=0.2058, late=0.3342, early_late_gap=-0.1283
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2676, early=0.2062, late=0.3339, early_late_gap=-0.1276
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2677, early=0.2062, late=0.3339, early_late_gap=-0.1277
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2674, early=0.2056, late=0.3339, early_late_gap=-0.1284
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-b471906-20260626/runs/gdn-d192-official-sudoku-nofs-reusefswt-s600-20260626T0855-b471906/output/futureseed_loop_case_seed52.html
