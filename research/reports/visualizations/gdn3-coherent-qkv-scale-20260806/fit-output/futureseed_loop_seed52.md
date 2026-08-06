# 9x9 GDN2 (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN2 (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=3.0120, total=3.0397, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=3.1116, loop_last_loss=3.0120, sec=93.2
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0991, early=0.1060, late=0.0743, early_late_gap=+0.0316
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=16.002, fs_decay=0.00, fs_raw_rms=0.078, fs_raw_rms_std=0.006, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.499:0.501, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1014, early=0.0993, late=0.0743, early_late_gap=+0.0250
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=16.002, fs_decay=0.00, fs_raw_rms=0.078, fs_raw_rms_std=0.007, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.499:0.501, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0991, early=0.0927, late=0.0743, early_late_gap=+0.0184
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=16.002, fs_decay=0.00, fs_raw_rms=0.078, fs_raw_rms_std=0.007, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.499:0.501, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0991, early=0.0927, late=0.0743, early_late_gap=+0.0184
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=16.002, fs_decay=0.00, fs_raw_rms=0.078, fs_raw_rms_std=0.007, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.499:0.501, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0991, early=0.0927, late=0.0743, early_late_gap=+0.0184
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=16.002, fs_decay=0.00, fs_raw_rms=0.078, fs_raw_rms_std=0.007, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.499:0.501, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Official Blank-Range Eval

- b51_55 (51-55, n=8): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1406, early=0.0966, late=0.1400, early_late_gap=-0.0434

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/gdn3-coherent-qkv-d256l12-fit-20260806T110533Z-ccd8897/output/futureseed_loop_case_seed52.html
