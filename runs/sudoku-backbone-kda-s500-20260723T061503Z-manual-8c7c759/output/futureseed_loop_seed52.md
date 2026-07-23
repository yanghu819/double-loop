# 9x9 KDA (official FLA) FutureSeed Loop Study

Mainline mechanism: KDA (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0065, total=1.0122, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0326, loop_last_loss=1.0065, sec=1.2
- loop 1: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.4881, early=0.4846, late=0.4922, early_late_gap=-0.0076
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=22.531, fs_decay=0.00, fs_raw_rms=0.249, fs_raw_rms_std=0.010, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0176, valid=0.0176, solved=0.0176, blank_acc=0.4999, early=0.4973, late=0.4983, early_late_gap=-0.0010
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=22.531, fs_decay=0.00, fs_raw_rms=0.242, fs_raw_rms_std=0.011, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0176, valid=0.0176, solved=0.0176, blank_acc=0.5013, early=0.4976, late=0.5013, early_late_gap=-0.0037
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=22.531, fs_decay=0.00, fs_raw_rms=0.240, fs_raw_rms_std=0.012, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0176, valid=0.0176, solved=0.0176, blank_acc=0.5018, early=0.4983, late=0.5017, early_late_gap=-0.0034
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=22.531, fs_decay=0.00, fs_raw_rms=0.239, fs_raw_rms_std=0.012, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0176, valid=0.0176, solved=0.0176, blank_acc=0.5018, early=0.4985, late=0.5017, early_late_gap=-0.0032
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=22.531, fs_decay=0.00, fs_raw_rms=0.239, fs_raw_rms_std=0.012, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-kda-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-kda-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.7539, blank_acc=0.9903, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-kda-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-kda-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5180, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-kda-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-kda-s500-20260723T061503Z-manual-8c7c759/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4668, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.7441, valid=0.7441, solved=0.7441, blank_acc=0.9905, early=0.9895, late=0.9904, early_late_gap=-0.0010
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5234, early=0.5217, late=0.5208, early_late_gap=+0.0009
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4687, early=0.4643, late=0.4659, early_late_gap=-0.0016

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-kda-s500-20260723T061503Z-manual-8c7c759/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/final-baseline-sparse-16f56e8/runs/sudoku-backbone-kda-s500-20260723T061503Z-manual-8c7c759/output/case_bank
