# 9x9 GDN2 (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN2 (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.5707, total=1.5796, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.6154, loop_last_loss=1.5707, sec=1311.3
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2642, early=0.2914, late=0.2266, early_late_gap=+0.0647
  future_seed: fs_gate=0.521, fs_update=1.000, fs_state_norm=16.663, fs_decay=0.00, fs_raw_rms=3.112, fs_raw_rms_std=0.212, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.284:0.718, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2885, early=0.3186, late=0.2422, early_late_gap=+0.0764
  future_seed: fs_gate=0.521, fs_update=1.000, fs_state_norm=16.663, fs_decay=0.00, fs_raw_rms=2.712, fs_raw_rms_std=0.213, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.284:0.718, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2897, early=0.3195, late=0.2428, early_late_gap=+0.0767
  future_seed: fs_gate=0.521, fs_update=1.000, fs_state_norm=16.663, fs_decay=0.00, fs_raw_rms=2.599, fs_raw_rms_std=0.213, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.284:0.718, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2900, early=0.3195, late=0.2428, early_late_gap=+0.0767
  future_seed: fs_gate=0.521, fs_update=1.000, fs_state_norm=16.663, fs_decay=0.00, fs_raw_rms=2.568, fs_raw_rms_std=0.212, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.284:0.718, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2907, early=0.3216, late=0.2432, early_late_gap=+0.0784
  future_seed: fs_gate=0.521, fs_update=1.000, fs_state_norm=16.663, fs_decay=0.00, fs_raw_rms=2.559, fs_raw_rms_std=0.212, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.284:0.718, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b51_55: index=/huyang2/double-loop/runs/gdn2-address-control-s9300-20260731T053946Z-3a54897/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/gdn2-address-control-s9300-20260731T053946Z-3a54897/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.2998, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 8}
- official_b56_60: index=/huyang2/double-loop/runs/gdn2-address-control-s9300-20260731T053946Z-3a54897/output/case_bank/official_b56_60/index.html; cases=/huyang2/double-loop/runs/gdn2-address-control-s9300-20260731T053946Z-3a54897/output/case_bank/official_b56_60/cases.json; final_loop=5, exact=0.0000, blank_acc=0.2803, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 8}
- official_b61_64: index=/huyang2/double-loop/runs/gdn2-address-control-s9300-20260731T053946Z-3a54897/output/case_bank/official_b61_64/index.html; cases=/huyang2/double-loop/runs/gdn2-address-control-s9300-20260731T053946Z-3a54897/output/case_bank/official_b61_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.2439, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 8}

## Official Blank-Range Eval

- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3035, early=0.3381, late=0.2524, early_late_gap=+0.0857
- b56_60 (56-60, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2781, early=0.3118, late=0.2377, early_late_gap=+0.0741
- b61_64 (61-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2444, early=0.2771, late=0.2012, early_late_gap=+0.0759

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/gdn2-address-control-s9300-20260731T053946Z-3a54897/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/gdn2-address-control-s9300-20260731T053946Z-3a54897/output/case_bank
