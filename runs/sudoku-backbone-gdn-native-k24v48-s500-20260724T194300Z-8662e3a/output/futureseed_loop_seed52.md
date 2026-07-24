# 9x9 GDN (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0598, total=1.0643, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0812, loop_last_loss=1.0598, sec=3029.5
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4740, early=0.4657, late=0.4793, early_late_gap=-0.0136
  future_seed: fs_gate=0.492, fs_update=1.000, fs_state_norm=16.703, fs_decay=0.00, fs_raw_rms=0.183, fs_raw_rms_std=0.010, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.4870, early=0.4814, late=0.4921, early_late_gap=-0.0107
  future_seed: fs_gate=0.492, fs_update=1.000, fs_state_norm=16.703, fs_decay=0.00, fs_raw_rms=0.179, fs_raw_rms_std=0.011, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0098, valid=0.0098, solved=0.0098, blank_acc=0.4874, early=0.4823, late=0.4916, early_late_gap=-0.0092
  future_seed: fs_gate=0.492, fs_update=1.000, fs_state_norm=16.703, fs_decay=0.00, fs_raw_rms=0.178, fs_raw_rms_std=0.011, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0117, valid=0.0117, solved=0.0117, blank_acc=0.4877, early=0.4816, late=0.4929, early_late_gap=-0.0113
  future_seed: fs_gate=0.492, fs_update=1.000, fs_state_norm=16.703, fs_decay=0.00, fs_raw_rms=0.177, fs_raw_rms_std=0.012, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0117, valid=0.0117, solved=0.0117, blank_acc=0.4876, early=0.4811, late=0.4929, early_late_gap=-0.0118
  future_seed: fs_gate=0.492, fs_update=1.000, fs_state_norm=16.703, fs_decay=0.00, fs_raw_rms=0.177, fs_raw_rms_std=0.012, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/runs/sudoku-backbone-gdn-native-k24v48-s500-20260724T194300Z-8662e3a/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-gdn-native-k24v48-s500-20260724T194300Z-8662e3a/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.4023, blank_acc=0.9708, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b51_55: index=/huyang2/double-loop/runs/sudoku-backbone-gdn-native-k24v48-s500-20260724T194300Z-8662e3a/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-gdn-native-k24v48-s500-20260724T194300Z-8662e3a/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5072, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/runs/sudoku-backbone-gdn-native-k24v48-s500-20260724T194300Z-8662e3a/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-gdn-native-k24v48-s500-20260724T194300Z-8662e3a/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4515, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.3887, valid=0.3887, solved=0.3887, blank_acc=0.9699, early=0.9682, late=0.9700, early_late_gap=-0.0017
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5085, early=0.5077, late=0.5034, early_late_gap=+0.0044
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4513, early=0.4536, late=0.4462, early_late_gap=+0.0074

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/sudoku-backbone-gdn-native-k24v48-s500-20260724T194300Z-8662e3a/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/sudoku-backbone-gdn-native-k24v48-s500-20260724T194300Z-8662e3a/output/case_bank
