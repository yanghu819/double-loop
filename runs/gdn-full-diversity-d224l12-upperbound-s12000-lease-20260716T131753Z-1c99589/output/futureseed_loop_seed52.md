# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.6176, total=0.7109, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9187, loop_last_loss=0.6176, sec=6770.6
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5781, early=0.5724, late=0.5822, early_late_gap=-0.0098
  future_seed: fs_gate=0.443, fs_update=1.000, fs_state_norm=14.176, fs_decay=0.00, fs_raw_rms=0.294, fs_raw_rms_std=0.030, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0664, valid=0.0664, solved=0.0664, blank_acc=0.6774, early=0.6732, late=0.6799, early_late_gap=-0.0067
  future_seed: fs_gate=0.443, fs_update=1.000, fs_state_norm=14.176, fs_decay=0.00, fs_raw_rms=0.262, fs_raw_rms_std=0.028, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.2617, valid=0.2617, solved=0.2617, blank_acc=0.7261, early=0.7217, late=0.7283, early_late_gap=-0.0065
  future_seed: fs_gate=0.443, fs_update=1.000, fs_state_norm=14.176, fs_decay=0.00, fs_raw_rms=0.291, fs_raw_rms_std=0.049, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.3496, valid=0.3496, solved=0.3496, blank_acc=0.7388, early=0.7378, late=0.7378, early_late_gap=+0.0000
  future_seed: fs_gate=0.443, fs_update=1.000, fs_state_norm=14.176, fs_decay=0.00, fs_raw_rms=0.332, fs_raw_rms_std=0.083, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.3730, valid=0.3730, solved=0.3730, blank_acc=0.7424, early=0.7425, late=0.7408, early_late_gap=+0.0017
  future_seed: fs_gate=0.443, fs_update=1.000, fs_state_norm=14.176, fs_decay=0.00, fs_raw_rms=0.348, fs_raw_rms_std=0.097, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale034-upperbound-1c99589-20260716/runs/gdn-full-diversity-d224l12-upperbound-s12000-lease-20260716T131753Z-1c99589/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-upperbound-1c99589-20260716/runs/gdn-full-diversity-d224l12-upperbound-s12000-lease-20260716T131753Z-1c99589/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 3, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale034-upperbound-1c99589-20260716/runs/gdn-full-diversity-d224l12-upperbound-s12000-lease-20260716T131753Z-1c99589/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-upperbound-1c99589-20260716/runs/gdn-full-diversity-d224l12-upperbound-s12000-lease-20260716T131753Z-1c99589/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.5195, blank_acc=0.8143, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale034-upperbound-1c99589-20260716/runs/gdn-full-diversity-d224l12-upperbound-s12000-lease-20260716T131753Z-1c99589/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-upperbound-1c99589-20260716/runs/gdn-full-diversity-d224l12-upperbound-s12000-lease-20260716T131753Z-1c99589/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.2383, blank_acc=0.6765, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.9980, valid=0.9980, solved=0.9980, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.5352, valid=0.5352, solved=0.5352, blank_acc=0.8243, early=0.8281, late=0.8254, early_late_gap=+0.0027
- b56_64 (56-64, n=512): exact=0.2285, valid=0.2285, solved=0.2285, blank_acc=0.6707, early=0.6681, late=0.6687, early_late_gap=-0.0006

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale034-upperbound-1c99589-20260716/runs/gdn-full-diversity-d224l12-upperbound-s12000-lease-20260716T131753Z-1c99589/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale034-upperbound-1c99589-20260716/runs/gdn-full-diversity-d224l12-upperbound-s12000-lease-20260716T131753Z-1c99589/output/case_bank
