# 9x9 GDN2 (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN2 (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9460, total=0.9754, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0541, loop_last_loss=0.9460, sec=1952.7
- loop 1: exact=0.0059, valid=0.0059, solved=0.0059, blank_acc=0.5008, early=0.5004, late=0.4991, early_late_gap=+0.0013
  future_seed: fs_gate=0.523, fs_update=1.000, fs_state_norm=16.721, fs_decay=0.00, fs_raw_rms=3.405, fs_raw_rms_std=0.185, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.278:0.708, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5354, early=0.5357, late=0.5336, early_late_gap=+0.0020
  future_seed: fs_gate=0.523, fs_update=1.000, fs_state_norm=16.721, fs_decay=0.00, fs_raw_rms=2.707, fs_raw_rms_std=0.260, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.278:0.708, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5484, early=0.5510, late=0.5459, early_late_gap=+0.0051
  future_seed: fs_gate=0.523, fs_update=1.000, fs_state_norm=16.721, fs_decay=0.00, fs_raw_rms=2.525, fs_raw_rms_std=0.316, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.278:0.708, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0254, valid=0.0254, solved=0.0254, blank_acc=0.5512, early=0.5539, late=0.5477, early_late_gap=+0.0062
  future_seed: fs_gate=0.523, fs_update=1.000, fs_state_norm=16.721, fs_decay=0.00, fs_raw_rms=2.458, fs_raw_rms_std=0.344, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.278:0.708, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0254, valid=0.0254, solved=0.0254, blank_acc=0.5515, early=0.5531, late=0.5486, early_late_gap=+0.0045
  future_seed: fs_gate=0.523, fs_update=1.000, fs_state_norm=16.721, fs_decay=0.00, fs_raw_rms=2.435, fs_raw_rms_std=0.352, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.278:0.708, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b51_55: index=/huyang2/double-loop/runs/gdn2-address-position_qk-s9600-20260731T062629Z-f8151df/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/gdn2-address-position_qk-s9600-20260731T062629Z-f8151df/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5507, selected={'solved_by_loop': 0, 'almost_solved': 7, 'hard_failure': 8}
- official_b56_60: index=/huyang2/double-loop/runs/gdn2-address-position_qk-s9600-20260731T062629Z-f8151df/output/case_bank/official_b56_60/index.html; cases=/huyang2/double-loop/runs/gdn2-address-position_qk-s9600-20260731T062629Z-f8151df/output/case_bank/official_b56_60/cases.json; final_loop=5, exact=0.0078, blank_acc=0.5232, selected={'solved_by_loop': 2, 'almost_solved': 4, 'hard_failure': 8}
- official_b61_64: index=/huyang2/double-loop/runs/gdn2-address-position_qk-s9600-20260731T062629Z-f8151df/output/case_bank/official_b61_64/index.html; cases=/huyang2/double-loop/runs/gdn2-address-position_qk-s9600-20260731T062629Z-f8151df/output/case_bank/official_b61_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.6314, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 8}

## Official Blank-Range Eval

- b51_55 (51-55, n=512): exact=0.0059, valid=0.0059, solved=0.0059, blank_acc=0.5845, early=0.5863, late=0.5903, early_late_gap=-0.0041
- b56_60 (56-60, n=512): exact=0.0059, valid=0.0059, solved=0.0059, blank_acc=0.5171, early=0.5193, late=0.5171, early_late_gap=+0.0022
- b61_64 (61-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6386, early=0.6317, late=0.6436, early_late_gap=-0.0119

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/gdn2-address-position_qk-s9600-20260731T062629Z-f8151df/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/gdn2-address-position_qk-s9600-20260731T062629Z-f8151df/output/case_bank
