# 9x9 KDA (official FLA) FutureSeed Loop Study

Mainline mechanism: KDA (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0179, total=1.0269, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0582, loop_last_loss=1.0179, sec=2805.8
- loop 1: exact=0.0020, valid=0.0020, solved=0.0020, blank_acc=0.4816, early=0.4721, late=0.4911, early_late_gap=-0.0191
  future_seed: fs_gate=0.494, fs_update=1.000, fs_state_norm=15.806, fs_decay=0.00, fs_raw_rms=0.240, fs_raw_rms_std=0.011, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.4999, early=0.4916, late=0.5073, early_late_gap=-0.0157
  future_seed: fs_gate=0.494, fs_update=1.000, fs_state_norm=15.806, fs_decay=0.00, fs_raw_rms=0.225, fs_raw_rms_std=0.013, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0176, valid=0.0176, solved=0.0176, blank_acc=0.5008, early=0.4924, late=0.5088, early_late_gap=-0.0164
  future_seed: fs_gate=0.494, fs_update=1.000, fs_state_norm=15.806, fs_decay=0.00, fs_raw_rms=0.221, fs_raw_rms_std=0.013, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0176, valid=0.0176, solved=0.0176, blank_acc=0.5009, early=0.4929, late=0.5078, early_late_gap=-0.0149
  future_seed: fs_gate=0.494, fs_update=1.000, fs_state_norm=15.806, fs_decay=0.00, fs_raw_rms=0.219, fs_raw_rms_std=0.013, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0176, valid=0.0176, solved=0.0176, blank_acc=0.5012, early=0.4928, late=0.5083, early_late_gap=-0.0156
  future_seed: fs_gate=0.494, fs_update=1.000, fs_state_norm=15.806, fs_decay=0.00, fs_raw_rms=0.219, fs_raw_rms_std=0.013, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/runs/sudoku-backbone-kda-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-kda-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.6328, blank_acc=0.9841, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b51_55: index=/huyang2/double-loop/runs/sudoku-backbone-kda-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-kda-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5210, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/runs/sudoku-backbone-kda-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-kda-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4683, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.6465, valid=0.6465, solved=0.6465, blank_acc=0.9854, early=0.9814, late=0.9898, early_late_gap=-0.0084
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5217, early=0.5230, late=0.5170, early_late_gap=+0.0060
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4664, early=0.4631, late=0.4684, early_late_gap=-0.0053

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/sudoku-backbone-kda-s500-sharedinit-20260724T154020Z-6e51f06/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/sudoku-backbone-kda-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank
