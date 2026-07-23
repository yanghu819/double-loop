# 9x9 GDN (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0001, total=1.0075, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0351, loop_last_loss=1.0001, sec=2482.3
- loop 1: exact=0.0137, valid=0.0137, solved=0.0137, blank_acc=0.4885, early=0.4834, late=0.4927, early_late_gap=-0.0093
  future_seed: fs_gate=0.489, fs_update=1.000, fs_state_norm=22.109, fs_decay=0.00, fs_raw_rms=0.136, fs_raw_rms_std=0.009, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5045, early=0.5007, late=0.5088, early_late_gap=-0.0080
  future_seed: fs_gate=0.489, fs_update=1.000, fs_state_norm=22.109, fs_decay=0.00, fs_raw_rms=0.135, fs_raw_rms_std=0.009, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5046, early=0.5014, late=0.5079, early_late_gap=-0.0066
  future_seed: fs_gate=0.489, fs_update=1.000, fs_state_norm=22.109, fs_decay=0.00, fs_raw_rms=0.135, fs_raw_rms_std=0.009, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5051, early=0.5016, late=0.5086, early_late_gap=-0.0070
  future_seed: fs_gate=0.489, fs_update=1.000, fs_state_norm=22.109, fs_decay=0.00, fs_raw_rms=0.135, fs_raw_rms_std=0.009, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5054, early=0.5022, late=0.5087, early_late_gap=-0.0065
  future_seed: fs_gate=0.489, fs_update=1.000, fs_state_norm=22.109, fs_decay=0.00, fs_raw_rms=0.135, fs_raw_rms_std=0.009, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.8711, blank_acc=0.9951, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 1}
- official_b51_55: index=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5291, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4739, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.8770, valid=0.8770, solved=0.8770, blank_acc=0.9952, early=0.9936, late=0.9952, early_late_gap=-0.0016
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5325, early=0.5311, late=0.5291, early_late_gap=+0.0021
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4738, early=0.4695, late=0.4749, early_late_gap=-0.0054

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn-s500-20260723T061503Z-manual-8c7c759/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-gdn-s500-20260723T061503Z-manual-8c7c759/output/case_bank
