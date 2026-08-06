# 9x9 GDN2 (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN2 (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=3.1327, total=3.1587, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=3.2270, loop_last_loss=3.1327, sec=95.7
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1149, early=0.1192, late=0.1419, early_late_gap=-0.0227
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=16.003, fs_decay=0.00, fs_raw_rms=0.069, fs_raw_rms_std=0.003, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.499:0.501, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, addr_scale=0.0000, addr_phase=0.0000, addr_qk_change=0.0000/0.0000, addr_norm_err=0.0e+00/0.0e+00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1194, early=0.1325, late=0.1419, early_late_gap=-0.0094
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=16.003, fs_decay=0.00, fs_raw_rms=0.069, fs_raw_rms_std=0.003, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.499:0.501, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, addr_scale=0.0000, addr_phase=0.0000, addr_qk_change=0.0000/0.0000, addr_norm_err=0.0e+00/0.0e+00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1171, early=0.1258, late=0.1419, early_late_gap=-0.0161
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=16.003, fs_decay=0.00, fs_raw_rms=0.069, fs_raw_rms_std=0.003, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.499:0.501, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, addr_scale=0.0000, addr_phase=0.0000, addr_qk_change=0.0000/0.0000, addr_norm_err=0.0e+00/0.0e+00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1149, early=0.1258, late=0.1419, early_late_gap=-0.0161
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=16.003, fs_decay=0.00, fs_raw_rms=0.069, fs_raw_rms_std=0.003, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.499:0.501, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, addr_scale=0.0000, addr_phase=0.0000, addr_qk_change=0.0000/0.0000, addr_norm_err=0.0e+00/0.0e+00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1149, early=0.1258, late=0.1419, early_late_gap=-0.0161
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=16.003, fs_decay=0.00, fs_raw_rms=0.069, fs_raw_rms_std=0.003, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.499:0.501, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, addr_scale=0.0000, addr_phase=0.0000, addr_qk_change=0.0000/0.0000, addr_norm_err=0.0e+00/0.0e+00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Official Blank-Range Eval

- b51_55 (51-55, n=8): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1267, early=0.1172, late=0.1200, early_late_gap=-0.0028

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/gdn3-position-qk-d256l12-fit-20260806T131002Z-9f2ee8d/output/futureseed_loop_case_seed52.html
