# 9x9 GDN2 (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN2 (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9661, total=0.9885, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0555, loop_last_loss=0.9661, sec=1272.8
- loop 1: exact=0.0020, valid=0.0020, solved=0.0020, blank_acc=0.4936, early=0.4956, late=0.4917, early_late_gap=+0.0039
  future_seed: fs_gate=0.522, fs_update=1.000, fs_state_norm=16.708, fs_decay=0.00, fs_raw_rms=3.162, fs_raw_rms_std=0.162, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.283:0.707, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5314, early=0.5301, late=0.5296, early_late_gap=+0.0005
  future_seed: fs_gate=0.522, fs_update=1.000, fs_state_norm=16.708, fs_decay=0.00, fs_raw_rms=2.582, fs_raw_rms_std=0.175, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.283:0.707, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5424, early=0.5385, late=0.5449, early_late_gap=-0.0064
  future_seed: fs_gate=0.522, fs_update=1.000, fs_state_norm=16.708, fs_decay=0.00, fs_raw_rms=2.445, fs_raw_rms_std=0.174, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.283:0.707, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5447, early=0.5424, late=0.5464, early_late_gap=-0.0041
  future_seed: fs_gate=0.522, fs_update=1.000, fs_state_norm=16.708, fs_decay=0.00, fs_raw_rms=2.397, fs_raw_rms_std=0.174, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.283:0.707, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5443, early=0.5418, late=0.5450, early_late_gap=-0.0031
  future_seed: fs_gate=0.522, fs_update=1.000, fs_state_norm=16.708, fs_decay=0.00, fs_raw_rms=2.383, fs_raw_rms_std=0.174, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.283:0.707, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b51_55: index=/huyang2/double-loop/runs/gdn2-address-position_qk-s9300-20260731T041227Z-eec018c/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/gdn2-address-position_qk-s9300-20260731T041227Z-eec018c/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5443, selected={'solved_by_loop': 0, 'almost_solved': 1, 'hard_failure': 8}
- official_b56_60: index=/huyang2/double-loop/runs/gdn2-address-position_qk-s9300-20260731T041227Z-eec018c/output/case_bank/official_b56_60/index.html; cases=/huyang2/double-loop/runs/gdn2-address-position_qk-s9300-20260731T041227Z-eec018c/output/case_bank/official_b56_60/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5068, selected={'solved_by_loop': 0, 'almost_solved': 4, 'hard_failure': 8}
- official_b61_64: index=/huyang2/double-loop/runs/gdn2-address-position_qk-s9300-20260731T041227Z-eec018c/output/case_bank/official_b61_64/index.html; cases=/huyang2/double-loop/runs/gdn2-address-position_qk-s9300-20260731T041227Z-eec018c/output/case_bank/official_b61_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5595, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 8}

## Official Blank-Range Eval

- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5660, early=0.5695, late=0.5620, early_late_gap=+0.0074
- b56_60 (56-60, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5057, early=0.5109, late=0.5035, early_late_gap=+0.0075
- b61_64 (61-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5594, early=0.5564, late=0.5604, early_late_gap=-0.0040

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/gdn2-address-position_qk-s9300-20260731T041227Z-eec018c/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/gdn2-address-position_qk-s9300-20260731T041227Z-eec018c/output/case_bank
