# 9x9 GDN2 (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN2 (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0186, total=1.0259, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0533, loop_last_loss=1.0186, sec=3020.5
- loop 1: exact=0.0098, valid=0.0098, solved=0.0098, blank_acc=0.4854, early=0.4825, late=0.4826, early_late_gap=-0.0001
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=15.963, fs_decay=0.00, fs_raw_rms=0.395, fs_raw_rms_std=0.027, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.456:0.572, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.5011, early=0.5000, late=0.5045, early_late_gap=-0.0045
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=15.963, fs_decay=0.00, fs_raw_rms=0.371, fs_raw_rms_std=0.030, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.456:0.572, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5011, early=0.4984, late=0.5063, early_late_gap=-0.0079
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=15.963, fs_decay=0.00, fs_raw_rms=0.362, fs_raw_rms_std=0.030, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.456:0.572, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5015, early=0.4995, late=0.5060, early_late_gap=-0.0066
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=15.963, fs_decay=0.00, fs_raw_rms=0.360, fs_raw_rms_std=0.030, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.456:0.572, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5012, early=0.4997, late=0.5057, early_late_gap=-0.0060
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=15.963, fs_decay=0.00, fs_raw_rms=0.359, fs_raw_rms_std=0.030, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.456:0.572, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-gdn2/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-gdn2/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.7871, blank_acc=0.9918, selected={'solved_by_loop': 8, 'almost_solved': 8, 'hard_failure': 2}
- official_b51_55: index=/huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-gdn2/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-gdn2/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5249, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 8}
- official_b56_64: index=/huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-gdn2/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-gdn2/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4709, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 8}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.7969, valid=0.7969, solved=0.7969, blank_acc=0.9918, early=0.9888, late=0.9926, early_late_gap=-0.0037
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5257, early=0.5220, late=0.5227, early_late_gap=-0.0007
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4680, early=0.4637, late=0.4633, early_late_gap=+0.0004

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-gdn2/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-gdn2/output/case_bank
