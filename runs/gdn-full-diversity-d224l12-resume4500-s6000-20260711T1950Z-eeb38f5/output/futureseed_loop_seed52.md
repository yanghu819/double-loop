# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.6097, total=0.6855, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.8743, loop_last_loss=0.6097, sec=4816.0
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5486, early=0.5409, late=0.5529, early_late_gap=-0.0120
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.781, fs_decay=0.00, fs_raw_rms=0.292, fs_raw_rms_std=0.027, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0273, valid=0.0273, solved=0.0273, blank_acc=0.6107, early=0.6069, late=0.6152, early_late_gap=-0.0083
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.781, fs_decay=0.00, fs_raw_rms=0.274, fs_raw_rms_std=0.029, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1738, valid=0.1738, solved=0.1738, blank_acc=0.6373, early=0.6397, late=0.6410, early_late_gap=-0.0012
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.781, fs_decay=0.00, fs_raw_rms=0.290, fs_raw_rms_std=0.043, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.2383, valid=0.2383, solved=0.2383, blank_acc=0.6431, early=0.6439, late=0.6458, early_late_gap=-0.0019
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.781, fs_decay=0.00, fs_raw_rms=0.305, fs_raw_rms_std=0.057, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.2500, valid=0.2500, solved=0.2500, blank_acc=0.6439, early=0.6444, late=0.6460, early_late_gap=-0.0016
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.781, fs_decay=0.00, fs_raw_rms=0.309, fs_raw_rms_std=0.061, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-resume4500-s6000-20260711T1950Z-eeb38f5/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-resume4500-s6000-20260711T1950Z-eeb38f5/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 4, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-resume4500-s6000-20260711T1950Z-eeb38f5/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-resume4500-s6000-20260711T1950Z-eeb38f5/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.4102, blank_acc=0.7724, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-resume4500-s6000-20260711T1950Z-eeb38f5/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-resume4500-s6000-20260711T1950Z-eeb38f5/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0781, blank_acc=0.5656, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.3926, valid=0.3926, solved=0.3926, blank_acc=0.7617, early=0.7630, late=0.7590, early_late_gap=+0.0040
- b56_64 (56-64, n=512): exact=0.1270, valid=0.1270, solved=0.1270, blank_acc=0.5625, early=0.5564, late=0.5568, early_late_gap=-0.0003

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-resume4500-s6000-20260711T1950Z-eeb38f5/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-resume4500-s6000-20260711T1950Z-eeb38f5/output/case_bank
