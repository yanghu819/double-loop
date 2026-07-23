# 9x9 GDN-Triton FutureSeed Loop Study

Mainline mechanism: GDN-Triton recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.5161, total=0.6533, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9546, loop_last_loss=0.5161, sec=5224.2
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5620, early=0.5598, late=0.5625, early_late_gap=-0.0027
  future_seed: fs_gate=0.406, fs_update=1.000, fs_state_norm=12.998, fs_decay=0.00, fs_raw_rms=0.331, fs_raw_rms_std=0.034, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0547, valid=0.0547, solved=0.0547, blank_acc=0.6804, early=0.6766, late=0.6809, early_late_gap=-0.0043
  future_seed: fs_gate=0.406, fs_update=1.000, fs_state_norm=12.998, fs_decay=0.00, fs_raw_rms=0.285, fs_raw_rms_std=0.032, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.2988, valid=0.2988, solved=0.2988, blank_acc=0.7370, early=0.7356, late=0.7380, early_late_gap=-0.0024
  future_seed: fs_gate=0.406, fs_update=1.000, fs_state_norm=12.998, fs_decay=0.00, fs_raw_rms=0.298, fs_raw_rms_std=0.047, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.4297, valid=0.4297, solved=0.4297, blank_acc=0.7575, early=0.7553, late=0.7560, early_late_gap=-0.0007
  future_seed: fs_gate=0.406, fs_update=1.000, fs_state_norm=12.993, fs_decay=0.00, fs_raw_rms=0.363, fs_raw_rms_std=0.084, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.4590, valid=0.4590, solved=0.4590, blank_acc=0.7634, early=0.7625, late=0.7630, early_late_gap=-0.0005
  future_seed: fs_gate=0.406, fs_update=1.000, fs_state_norm=12.985, fs_decay=0.00, fs_raw_rms=0.385, fs_raw_rms_std=0.104, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale036-formal-4ef23df-20260723/runs/gdn-data-coverage-control-s31500-20260723T1030Z-4ef23df/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale036-formal-4ef23df-20260723/runs/gdn-data-coverage-control-s31500-20260723T1030Z-4ef23df/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 4, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale036-formal-4ef23df-20260723/runs/gdn-data-coverage-control-s31500-20260723T1030Z-4ef23df/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale036-formal-4ef23df-20260723/runs/gdn-data-coverage-control-s31500-20260723T1030Z-4ef23df/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.5898, blank_acc=0.8444, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale036-formal-4ef23df-20260723/runs/gdn-data-coverage-control-s31500-20260723T1030Z-4ef23df/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale036-formal-4ef23df-20260723/runs/gdn-data-coverage-control-s31500-20260723T1030Z-4ef23df/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.3086, blank_acc=0.6952, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.6094, valid=0.6094, solved=0.6094, blank_acc=0.8371, early=0.8383, late=0.8358, early_late_gap=+0.0025
- b56_64 (56-64, n=512): exact=0.3086, valid=0.3086, solved=0.3086, blank_acc=0.6858, early=0.6839, late=0.6859, early_late_gap=-0.0020

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale036-formal-4ef23df-20260723/runs/gdn-data-coverage-control-s31500-20260723T1030Z-4ef23df/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale036-formal-4ef23df-20260723/runs/gdn-data-coverage-control-s31500-20260723T1030Z-4ef23df/output/case_bank
