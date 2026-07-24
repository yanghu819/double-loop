# 9x9 GDN (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0509, total=1.0552, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0697, loop_last_loss=1.0509, sec=2457.0
- loop 1: exact=0.0020, valid=0.0020, solved=0.0020, blank_acc=0.4751, early=0.4688, late=0.4773, early_late_gap=-0.0084
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=15.601, fs_decay=0.00, fs_raw_rms=0.164, fs_raw_rms_std=0.009, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0059, valid=0.0059, solved=0.0059, blank_acc=0.4873, early=0.4851, late=0.4889, early_late_gap=-0.0038
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=15.601, fs_decay=0.00, fs_raw_rms=0.159, fs_raw_rms_std=0.010, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0059, valid=0.0059, solved=0.0059, blank_acc=0.4880, early=0.4865, late=0.4889, early_late_gap=-0.0024
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=15.601, fs_decay=0.00, fs_raw_rms=0.158, fs_raw_rms_std=0.010, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0059, valid=0.0059, solved=0.0059, blank_acc=0.4888, early=0.4870, late=0.4899, early_late_gap=-0.0028
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=15.601, fs_decay=0.00, fs_raw_rms=0.157, fs_raw_rms_std=0.010, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0059, valid=0.0059, solved=0.0059, blank_acc=0.4887, early=0.4871, late=0.4898, early_late_gap=-0.0026
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=15.601, fs_decay=0.00, fs_raw_rms=0.157, fs_raw_rms_std=0.010, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/runs/sudoku-backbone-gdn-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-gdn-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.3867, blank_acc=0.9686, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b51_55: index=/huyang2/double-loop/runs/sudoku-backbone-gdn-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-gdn-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5087, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/runs/sudoku-backbone-gdn-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-gdn-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4571, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.3633, valid=0.3633, solved=0.3633, blank_acc=0.9701, early=0.9635, late=0.9744, early_late_gap=-0.0109
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5150, early=0.5116, late=0.5135, early_late_gap=-0.0020
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4541, early=0.4562, late=0.4465, early_late_gap=+0.0097

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/sudoku-backbone-gdn-s500-sharedinit-20260724T154020Z-6e51f06/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/sudoku-backbone-gdn-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank
