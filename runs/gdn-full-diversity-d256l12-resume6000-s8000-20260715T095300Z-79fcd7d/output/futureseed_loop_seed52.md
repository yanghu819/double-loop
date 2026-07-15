# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.7586, total=0.8434, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0296, loop_last_loss=0.7586, sec=6563.2
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5402, early=0.5408, late=0.5418, early_late_gap=-0.0010
  future_seed: fs_gate=0.436, fs_update=1.000, fs_state_norm=13.962, fs_decay=0.00, fs_raw_rms=0.322, fs_raw_rms_std=0.036, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0254, valid=0.0254, solved=0.0254, blank_acc=0.6219, early=0.6139, late=0.6235, early_late_gap=-0.0097
  future_seed: fs_gate=0.436, fs_update=1.000, fs_state_norm=13.962, fs_decay=0.00, fs_raw_rms=0.299, fs_raw_rms_std=0.038, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1426, valid=0.1426, solved=0.1426, blank_acc=0.6663, early=0.6630, late=0.6651, early_late_gap=-0.0021
  future_seed: fs_gate=0.436, fs_update=1.000, fs_state_norm=13.962, fs_decay=0.00, fs_raw_rms=0.316, fs_raw_rms_std=0.045, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.2441, valid=0.2441, solved=0.2441, blank_acc=0.6802, early=0.6778, late=0.6788, early_late_gap=-0.0011
  future_seed: fs_gate=0.436, fs_update=1.000, fs_state_norm=13.962, fs_decay=0.00, fs_raw_rms=0.340, fs_raw_rms_std=0.067, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.2539, valid=0.2539, solved=0.2539, blank_acc=0.6821, early=0.6793, late=0.6811, early_late_gap=-0.0019
  future_seed: fs_gate=0.436, fs_update=1.000, fs_state_norm=13.962, fs_decay=0.00, fs_raw_rms=0.348, fs_raw_rms_std=0.075, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale032-d256-formal-79fcd7d-20260712/runs/gdn-full-diversity-d256l12-resume6000-s8000-20260715T095300Z-79fcd7d/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale032-d256-formal-79fcd7d-20260712/runs/gdn-full-diversity-d256l12-resume6000-s8000-20260715T095300Z-79fcd7d/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 4, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale032-d256-formal-79fcd7d-20260712/runs/gdn-full-diversity-d256l12-resume6000-s8000-20260715T095300Z-79fcd7d/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale032-d256-formal-79fcd7d-20260712/runs/gdn-full-diversity-d256l12-resume6000-s8000-20260715T095300Z-79fcd7d/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.4336, blank_acc=0.7981, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale032-d256-formal-79fcd7d-20260712/runs/gdn-full-diversity-d256l12-resume6000-s8000-20260715T095300Z-79fcd7d/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale032-d256-formal-79fcd7d-20260712/runs/gdn-full-diversity-d256l12-resume6000-s8000-20260715T095300Z-79fcd7d/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.1328, blank_acc=0.6210, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.4219, valid=0.4219, solved=0.4219, blank_acc=0.7773, early=0.7732, late=0.7783, early_late_gap=-0.0052
- b56_64 (56-64, n=512): exact=0.1562, valid=0.1562, solved=0.1562, blank_acc=0.6115, early=0.6123, late=0.6101, early_late_gap=+0.0021

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale032-d256-formal-79fcd7d-20260712/runs/gdn-full-diversity-d256l12-resume6000-s8000-20260715T095300Z-79fcd7d/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale032-d256-formal-79fcd7d-20260712/runs/gdn-full-diversity-d256l12-resume6000-s8000-20260715T095300Z-79fcd7d/output/case_bank
