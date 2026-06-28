# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9903, total=0.9903, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.2650, loop_last_loss=0.9903, sec=2703.2
- loop 1: exact=0.0029, valid=0.0029, solved=0.0029, blank_acc=0.4503, early=0.4531, late=0.4511, early_late_gap=+0.0019
  future_seed: fs_gate=0.482, fs_update=1.000, fs_state_norm=7.720, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0278, valid=0.0278, solved=0.0278, blank_acc=0.4997, early=0.4997, late=0.5009, early_late_gap=-0.0012
  future_seed: fs_gate=0.482, fs_update=1.000, fs_state_norm=7.720, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5392, early=0.5366, late=0.5428, early_late_gap=-0.0062
  future_seed: fs_gate=0.482, fs_update=1.000, fs_state_norm=7.720, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0303, valid=0.0303, solved=0.0303, blank_acc=0.5454, early=0.5438, late=0.5482, early_late_gap=-0.0044
  future_seed: fs_gate=0.482, fs_update=1.000, fs_state_norm=7.720, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0312, valid=0.0312, solved=0.0312, blank_acc=0.5462, early=0.5443, late=0.5490, early_late_gap=-0.0047
  future_seed: fs_gate=0.482, fs_update=1.000, fs_state_norm=7.720, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-hardcurr-s4000-d5658e0-20260628/runs/gdn-d192-official-sudoku-fs-hardcurr-s4000-20260628T1435Z-d5658e0/output/futureseed_loop_case_seed52.html
