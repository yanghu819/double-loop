# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.7939, total=0.8355, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9706, loop_last_loss=0.7939, sec=0.6
- loop 1: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5278, early=0.5212, late=0.5356, early_late_gap=-0.0144
  future_seed: fs_gate=0.479, fs_update=1.000, fs_state_norm=15.323, fs_decay=0.00, fs_raw_rms=0.273, fs_raw_rms_std=0.024, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5725, early=0.5691, late=0.5785, early_late_gap=-0.0094
  future_seed: fs_gate=0.479, fs_update=1.000, fs_state_norm=15.323, fs_decay=0.00, fs_raw_rms=0.278, fs_raw_rms_std=0.030, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0312, valid=0.0312, solved=0.0312, blank_acc=0.5845, early=0.5758, late=0.5946, early_late_gap=-0.0188
  future_seed: fs_gate=0.479, fs_update=1.000, fs_state_norm=15.323, fs_decay=0.00, fs_raw_rms=0.285, fs_raw_rms_std=0.035, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0312, valid=0.0312, solved=0.0312, blank_acc=0.5858, early=0.5782, late=0.5951, early_late_gap=-0.0169
  future_seed: fs_gate=0.479, fs_update=1.000, fs_state_norm=15.323, fs_decay=0.00, fs_raw_rms=0.288, fs_raw_rms_std=0.036, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0312, valid=0.0312, solved=0.0312, blank_acc=0.5864, early=0.5790, late=0.5954, early_late_gap=-0.0165
  future_seed: fs_gate=0.479, fs_update=1.000, fs_state_norm=15.323, fs_decay=0.00, fs_raw_rms=0.288, fs_raw_rms_std=0.037, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-fs-step3000-evalviz-20260715T1610Z-eeb38f5/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-fs-step3000-evalviz-20260715T1610Z-eeb38f5/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 4, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-fs-step3000-evalviz-20260715T1610Z-eeb38f5/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-fs-step3000-evalviz-20260715T1610Z-eeb38f5/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0039, blank_acc=0.6540, selected={'solved_by_loop': 1, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-fs-step3000-evalviz-20260715T1610Z-eeb38f5/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-fs-step3000-evalviz-20260715T1610Z-eeb38f5/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5257, selected={'solved_by_loop': 0, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.0117, valid=0.0117, solved=0.0117, blank_acc=0.6560, early=0.6505, late=0.6567, early_late_gap=-0.0062
- b56_64 (56-64, n=512): exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.5270, early=0.5233, late=0.5241, early_late_gap=-0.0007

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-fs-step3000-evalviz-20260715T1610Z-eeb38f5/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-fs-step3000-evalviz-20260715T1610Z-eeb38f5/output/case_bank
