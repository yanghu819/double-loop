# 9x9 GDN2 (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN2 (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.7102, total=0.7778, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9450, loop_last_loss=0.7102, sec=88991.9
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5482, early=0.5440, late=0.5476, early_late_gap=-0.0036
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=15.928, fs_decay=0.00, fs_raw_rms=10.860, fs_raw_rms_std=1.013, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.011:0.746, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, addr_scale=0.0000, addr_phase=0.0000, addr_qk_change=0.5452/0.9671, addr_norm_err=5.2e+01/5.1e+01, gdn3_shared_addr=0.1197/6.8103/3.9260, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0352, valid=0.0352, solved=0.0352, blank_acc=0.6210, early=0.6137, late=0.6229, early_late_gap=-0.0092
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=15.928, fs_decay=0.00, fs_raw_rms=9.728, fs_raw_rms_std=1.257, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.011:0.746, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, addr_scale=0.0000, addr_phase=0.0000, addr_qk_change=0.5648/0.9835, addr_norm_err=5.3e+01/5.1e+01, gdn3_shared_addr=0.1197/6.8103/3.9260, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1562, valid=0.1562, solved=0.1562, blank_acc=0.6569, early=0.6479, late=0.6606, early_late_gap=-0.0127
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=15.928, fs_decay=0.00, fs_raw_rms=8.400, fs_raw_rms_std=1.583, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.011:0.746, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, addr_scale=0.0000, addr_phase=0.0000, addr_qk_change=0.5978/1.0120, addr_norm_err=5.3e+01/5.2e+01, gdn3_shared_addr=0.1197/6.8103/3.9260, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.2051, valid=0.2051, solved=0.2051, blank_acc=0.6643, early=0.6548, late=0.6690, early_late_gap=-0.0142
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=15.928, fs_decay=0.00, fs_raw_rms=8.009, fs_raw_rms_std=1.695, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.011:0.746, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, addr_scale=0.0000, addr_phase=0.0000, addr_qk_change=0.6098/1.0224, addr_norm_err=5.4e+01/5.3e+01, gdn3_shared_addr=0.1197/6.8103/3.9260, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.2168, valid=0.2168, solved=0.2168, blank_acc=0.6658, early=0.6563, late=0.6707, early_late_gap=-0.0143
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=15.928, fs_decay=0.00, fs_raw_rms=7.906, fs_raw_rms_std=1.714, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.011:0.746, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, addr_scale=0.0000, addr_phase=0.0000, addr_qk_change=0.6130/1.0255, addr_norm_err=5.4e+01/5.3e+01, gdn3_shared_addr=0.1197/6.8103/3.9260, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/runs/gdn3-shared-namespace-d256l12-s12000-20260805T044918Z-a68c683/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/runs/gdn3-shared-namespace-d256l12-s12000-20260805T044918Z-a68c683/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 2, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/runs/gdn3-shared-namespace-d256l12-s12000-20260805T044918Z-a68c683/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/gdn3-shared-namespace-d256l12-s12000-20260805T044918Z-a68c683/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.2695, blank_acc=0.7505, selected={'solved_by_loop': 8, 'almost_solved': 8, 'hard_failure': 8}
- official_b56_60: index=/huyang2/double-loop/runs/gdn3-shared-namespace-d256l12-s12000-20260805T044918Z-a68c683/output/case_bank/official_b56_60/index.html; cases=/huyang2/double-loop/runs/gdn3-shared-namespace-d256l12-s12000-20260805T044918Z-a68c683/output/case_bank/official_b56_60/cases.json; final_loop=5, exact=0.1289, blank_acc=0.5927, selected={'solved_by_loop': 8, 'almost_solved': 8, 'hard_failure': 8}
- official_b61_64: index=/huyang2/double-loop/runs/gdn3-shared-namespace-d256l12-s12000-20260805T044918Z-a68c683/output/case_bank/official_b61_64/index.html; cases=/huyang2/double-loop/runs/gdn3-shared-namespace-d256l12-s12000-20260805T044918Z-a68c683/output/case_bank/official_b61_64/cases.json; final_loop=5, exact=0.1016, blank_acc=0.7505, selected={'solved_by_loop': 8, 'almost_solved': 8, 'hard_failure': 8}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.3008, valid=0.3008, solved=0.3008, blank_acc=0.7432, early=0.7400, late=0.7465, early_late_gap=-0.0065
- b56_60 (56-60, n=512): exact=0.0996, valid=0.0996, solved=0.0996, blank_acc=0.5831, early=0.5771, late=0.5910, early_late_gap=-0.0139
- b61_64 (61-64, n=512): exact=0.0762, valid=0.0762, solved=0.0762, blank_acc=0.7562, early=0.7548, late=0.7624, early_late_gap=-0.0076

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/gdn3-shared-namespace-d256l12-s12000-20260805T044918Z-a68c683/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/gdn3-shared-namespace-d256l12-s12000-20260805T044918Z-a68c683/output/case_bank
