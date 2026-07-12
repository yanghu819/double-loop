# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.7205, total=0.7764, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9851, loop_last_loss=0.7205, sec=0.5
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5546, early=0.5523, late=0.5603, early_late_gap=-0.0080
  future_seed: fs_gate=0.455, fs_update=1.000, fs_state_norm=14.556, fs_decay=0.00, fs_raw_rms=0.294, fs_raw_rms_std=0.029, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0273, valid=0.0273, solved=0.0273, blank_acc=0.6412, early=0.6398, late=0.6465, early_late_gap=-0.0067
  future_seed: fs_gate=0.455, fs_update=1.000, fs_state_norm=14.556, fs_decay=0.00, fs_raw_rms=0.270, fs_raw_rms_std=0.029, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1797, valid=0.1797, solved=0.1797, blank_acc=0.6829, early=0.6800, late=0.6845, early_late_gap=-0.0045
  future_seed: fs_gate=0.455, fs_update=1.000, fs_state_norm=14.556, fs_decay=0.00, fs_raw_rms=0.291, fs_raw_rms_std=0.041, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.2617, valid=0.2617, solved=0.2617, blank_acc=0.6966, early=0.6962, late=0.6971, early_late_gap=-0.0009
  future_seed: fs_gate=0.455, fs_update=1.000, fs_state_norm=14.556, fs_decay=0.00, fs_raw_rms=0.314, fs_raw_rms_std=0.066, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.2812, valid=0.2812, solved=0.2812, blank_acc=0.6972, early=0.6966, late=0.6985, early_late_gap=-0.0018
  future_seed: fs_gate=0.455, fs_update=1.000, fs_state_norm=14.556, fs_decay=0.00, fs_raw_rms=0.321, fs_raw_rms_std=0.075, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.2852, valid=0.2852, solved=0.2852, blank_acc=0.6972, early=0.6976, late=0.6982, early_late_gap=-0.0007
  future_seed: fs_gate=0.455, fs_update=1.000, fs_state_norm=14.555, fs_decay=0.00, fs_raw_rms=0.323, fs_raw_rms_std=0.077, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 7: exact=0.2871, valid=0.2871, solved=0.2871, blank_acc=0.6975, early=0.6978, late=0.6984, early_late_gap=-0.0006
  future_seed: fs_gate=0.455, fs_update=1.000, fs_state_norm=14.555, fs_decay=0.00, fs_raw_rms=0.323, fs_raw_rms_std=0.077, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 8: exact=0.2871, valid=0.2871, solved=0.2871, blank_acc=0.6975, early=0.6977, late=0.6982, early_late_gap=-0.0006
  future_seed: fs_gate=0.455, fs_update=1.000, fs_state_norm=14.555, fs_decay=0.00, fs_raw_rms=0.323, fs_raw_rms_std=0.078, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale031-loop8-77481aa-20260712T1943/runs/gdn-full-diversity-loop8-step8200-evalviz-20260712T121529Z-77481aa/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale031-loop8-77481aa-20260712T1943/runs/gdn-full-diversity-loop8-step8200-evalviz-20260712T121529Z-77481aa/output/case_bank/official_b46_50/cases.json; final_loop=8, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 3, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale031-loop8-77481aa-20260712T1943/runs/gdn-full-diversity-loop8-step8200-evalviz-20260712T121529Z-77481aa/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale031-loop8-77481aa-20260712T1943/runs/gdn-full-diversity-loop8-step8200-evalviz-20260712T121529Z-77481aa/output/case_bank/official_b51_55/cases.json; final_loop=8, exact=0.4414, blank_acc=0.7929, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale031-loop8-77481aa-20260712T1943/runs/gdn-full-diversity-loop8-step8200-evalviz-20260712T121529Z-77481aa/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale031-loop8-77481aa-20260712T1943/runs/gdn-full-diversity-loop8-step8200-evalviz-20260712T121529Z-77481aa/output/case_bank/official_b56_64/cases.json; final_loop=8, exact=0.1094, blank_acc=0.6220, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.4180, valid=0.4180, solved=0.4180, blank_acc=0.7798, early=0.7774, late=0.7800, early_late_gap=-0.0027
- b56_64 (56-64, n=512): exact=0.1484, valid=0.1484, solved=0.1484, blank_acc=0.6322, early=0.6310, late=0.6281, early_late_gap=+0.0029

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale031-loop8-77481aa-20260712T1943/runs/gdn-full-diversity-loop8-step8200-evalviz-20260712T121529Z-77481aa/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale031-loop8-77481aa-20260712T1943/runs/gdn-full-diversity-loop8-step8200-evalviz-20260712T121529Z-77481aa/output/case_bank
