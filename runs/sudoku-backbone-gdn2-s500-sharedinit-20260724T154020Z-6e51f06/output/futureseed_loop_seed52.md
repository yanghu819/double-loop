# 9x9 GDN2 (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN2 (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0062, total=1.0133, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0381, loop_last_loss=1.0062, sec=2945.1
- loop 1: exact=0.0137, valid=0.0137, solved=0.0137, blank_acc=0.4885, early=0.4849, late=0.4865, early_late_gap=-0.0016
  future_seed: fs_gate=0.497, fs_update=1.000, fs_state_norm=15.918, fs_decay=0.00, fs_raw_rms=0.405, fs_raw_rms_std=0.028, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5016, early=0.4973, late=0.5023, early_late_gap=-0.0050
  future_seed: fs_gate=0.497, fs_update=1.000, fs_state_norm=15.918, fs_decay=0.00, fs_raw_rms=0.382, fs_raw_rms_std=0.030, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5009, early=0.4957, late=0.5008, early_late_gap=-0.0051
  future_seed: fs_gate=0.497, fs_update=1.000, fs_state_norm=15.918, fs_decay=0.00, fs_raw_rms=0.375, fs_raw_rms_std=0.030, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5012, early=0.4955, late=0.5024, early_late_gap=-0.0069
  future_seed: fs_gate=0.497, fs_update=1.000, fs_state_norm=15.918, fs_decay=0.00, fs_raw_rms=0.373, fs_raw_rms_std=0.030, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5011, early=0.4951, late=0.5022, early_late_gap=-0.0071
  future_seed: fs_gate=0.497, fs_update=1.000, fs_state_norm=15.918, fs_decay=0.00, fs_raw_rms=0.372, fs_raw_rms_std=0.030, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/runs/sudoku-backbone-gdn2-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-gdn2-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.7930, blank_acc=0.9923, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 1}
- official_b51_55: index=/huyang2/double-loop/runs/sudoku-backbone-gdn2-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-gdn2-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5167, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/runs/sudoku-backbone-gdn2-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-gdn2-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4726, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.7969, valid=0.7969, solved=0.7969, blank_acc=0.9924, early=0.9902, late=0.9931, early_late_gap=-0.0028
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5259, early=0.5224, late=0.5249, early_late_gap=-0.0025
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4722, early=0.4666, late=0.4719, early_late_gap=-0.0054

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/sudoku-backbone-gdn2-s500-sharedinit-20260724T154020Z-6e51f06/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/sudoku-backbone-gdn2-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank
