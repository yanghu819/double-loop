# 9x9 Raven (official FLA) FutureSeed Loop Study

Mainline mechanism: Raven (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.1388, total=1.1494, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.1819, loop_last_loss=1.1388, sec=3851.7
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4197, early=0.4055, late=0.4358, early_late_gap=-0.0304
  future_seed: fs_gate=0.490, fs_update=1.000, fs_state_norm=15.674, fs_decay=0.00, fs_raw_rms=0.534, fs_raw_rms_std=0.014, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.427:0.567, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4295, early=0.4159, late=0.4447, early_late_gap=-0.0287
  future_seed: fs_gate=0.490, fs_update=1.000, fs_state_norm=15.674, fs_decay=0.00, fs_raw_rms=0.527, fs_raw_rms_std=0.014, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.427:0.567, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4287, early=0.4159, late=0.4425, early_late_gap=-0.0265
  future_seed: fs_gate=0.490, fs_update=1.000, fs_state_norm=15.674, fs_decay=0.00, fs_raw_rms=0.524, fs_raw_rms_std=0.014, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.427:0.567, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4291, early=0.4168, late=0.4427, early_late_gap=-0.0259
  future_seed: fs_gate=0.490, fs_update=1.000, fs_state_norm=15.674, fs_decay=0.00, fs_raw_rms=0.523, fs_raw_rms_std=0.014, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.427:0.567, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4292, early=0.4172, late=0.4419, early_late_gap=-0.0248
  future_seed: fs_gate=0.490, fs_update=1.000, fs_state_norm=15.674, fs_decay=0.00, fs_raw_rms=0.523, fs_raw_rms_std=0.014, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fs2_delta=0.0000, fs2_gate_std=0.0000, fs2_batch_std=0.0000, fs2_feature_std=0.0000, fs2_gate_range=0.427:0.567, fs2_seed_change=0.0000, fs2_block_active=0.0, fs2_block_raw_norm=0.000, slow_decay=0, slow_rho/lag=0.000/0.000, slow_tv=0.0000->0.0000, slow_change=0.0000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-raven/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-raven/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.0000, blank_acc=0.7733, selected={'solved_by_loop': 0, 'almost_solved': 8, 'hard_failure': 8}
- official_b51_55: index=/huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-raven/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-raven/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4427, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 8}
- official_b56_64: index=/huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-raven/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-raven/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4082, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 8}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.7771, early=0.7636, late=0.8040, early_late_gap=-0.0405
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4475, early=0.4359, late=0.4567, early_late_gap=-0.0208
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4052, early=0.3964, late=0.4170, early_late_gap=-0.0206

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-raven/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/raven-futureseed-vs-gdn2-20260801T104903Z-f2306f8-raven/output/case_bank
