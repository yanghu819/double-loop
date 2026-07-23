# 9x9 KDA (official FLA) FutureSeed Loop Study

Mainline mechanism: KDA (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9642, total=0.9717, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9976, loop_last_loss=0.9642, sec=102.6
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5034, early=0.5093, late=0.5033, early_late_gap=+0.0059
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=22.532, fs_decay=0.00, fs_raw_rms=0.248, fs_raw_rms_std=0.010, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5179, early=0.5346, late=0.4917, early_late_gap=+0.0429
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=22.532, fs_decay=0.00, fs_raw_rms=0.241, fs_raw_rms_std=0.011, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5224, early=0.5312, late=0.5167, early_late_gap=+0.0145
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=22.532, fs_decay=0.00, fs_raw_rms=0.239, fs_raw_rms_std=0.011, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5202, early=0.5312, late=0.5117, early_late_gap=+0.0195
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=22.532, fs_decay=0.00, fs_raw_rms=0.238, fs_raw_rms_std=0.011, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5218, early=0.5329, late=0.5100, early_late_gap=+0.0229
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=22.532, fs_decay=0.00, fs_raw_rms=0.238, fs_raw_rms_std=0.011, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-kda-memory-probe-s501-20260723T0920Z-8c7c759/output/futureseed_loop_case_seed52.html
