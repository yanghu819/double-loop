# 9x9 GDN2 (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN2 (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0114, total=1.0175, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0389, loop_last_loss=1.0114, sec=2766.9
- loop 1: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.4904, early=0.4833, late=0.4949, early_late_gap=-0.0116
  future_seed: fs_gate=0.504, fs_update=1.000, fs_state_norm=22.789, fs_decay=0.00, fs_raw_rms=0.469, fs_raw_rms_std=0.032, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0176, valid=0.0176, solved=0.0176, blank_acc=0.5007, early=0.4952, late=0.5065, early_late_gap=-0.0113
  future_seed: fs_gate=0.504, fs_update=1.000, fs_state_norm=22.789, fs_decay=0.00, fs_raw_rms=0.450, fs_raw_rms_std=0.033, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5016, early=0.4958, late=0.5079, early_late_gap=-0.0121
  future_seed: fs_gate=0.504, fs_update=1.000, fs_state_norm=22.789, fs_decay=0.00, fs_raw_rms=0.443, fs_raw_rms_std=0.033, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5012, early=0.4973, late=0.5067, early_late_gap=-0.0094
  future_seed: fs_gate=0.504, fs_update=1.000, fs_state_norm=22.789, fs_decay=0.00, fs_raw_rms=0.442, fs_raw_rms_std=0.033, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5013, early=0.4971, late=0.5070, early_late_gap=-0.0099
  future_seed: fs_gate=0.504, fs_update=1.000, fs_state_norm=22.789, fs_decay=0.00, fs_raw_rms=0.441, fs_raw_rms_std=0.033, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn2-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn2-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.7539, blank_acc=0.9907, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 3}
- official_b51_55: index=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn2-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn2-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5231, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn2-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn2-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4697, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.7754, valid=0.7754, solved=0.7754, blank_acc=0.9916, early=0.9891, late=0.9927, early_late_gap=-0.0036
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5199, early=0.5167, late=0.5179, early_late_gap=-0.0012
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4712, early=0.4657, late=0.4704, early_late_gap=-0.0046

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn2-s500-20260723T061503Z-manual-8c7c759/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn2-s500-20260723T061503Z-manual-8c7c759/output/case_bank
