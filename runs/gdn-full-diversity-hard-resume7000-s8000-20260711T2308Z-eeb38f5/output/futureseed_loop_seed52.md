# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.7353, total=0.8115, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9849, loop_last_loss=0.7353, sec=3324.4
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5605, early=0.5539, late=0.5653, early_late_gap=-0.0115
  future_seed: fs_gate=0.456, fs_update=1.000, fs_state_norm=14.581, fs_decay=0.00, fs_raw_rms=0.288, fs_raw_rms_std=0.027, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0508, valid=0.0508, solved=0.0508, blank_acc=0.6534, early=0.6469, late=0.6540, early_late_gap=-0.0070
  future_seed: fs_gate=0.456, fs_update=1.000, fs_state_norm=14.581, fs_decay=0.00, fs_raw_rms=0.267, fs_raw_rms_std=0.028, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1875, valid=0.1875, solved=0.1875, blank_acc=0.6938, early=0.6903, late=0.6961, early_late_gap=-0.0058
  future_seed: fs_gate=0.456, fs_update=1.000, fs_state_norm=14.581, fs_decay=0.00, fs_raw_rms=0.288, fs_raw_rms_std=0.039, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.2598, valid=0.2598, solved=0.2598, blank_acc=0.7049, early=0.7028, late=0.7052, early_late_gap=-0.0024
  future_seed: fs_gate=0.456, fs_update=1.000, fs_state_norm=14.581, fs_decay=0.00, fs_raw_rms=0.311, fs_raw_rms_std=0.062, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.2715, valid=0.2715, solved=0.2715, blank_acc=0.7055, early=0.7037, late=0.7049, early_late_gap=-0.0011
  future_seed: fs_gate=0.456, fs_update=1.000, fs_state_norm=14.581, fs_decay=0.00, fs_raw_rms=0.318, fs_raw_rms_std=0.070, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale029-full-diversity-eeb38f5-20260711T1505Z/runs/gdn-full-diversity-hard-resume7000-s8000-20260711T2308Z-eeb38f5/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale029-full-diversity-eeb38f5-20260711T1505Z/runs/gdn-full-diversity-hard-resume7000-s8000-20260711T2308Z-eeb38f5/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 3, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale029-full-diversity-eeb38f5-20260711T1505Z/runs/gdn-full-diversity-hard-resume7000-s8000-20260711T2308Z-eeb38f5/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale029-full-diversity-eeb38f5-20260711T1505Z/runs/gdn-full-diversity-hard-resume7000-s8000-20260711T2308Z-eeb38f5/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.4375, blank_acc=0.7963, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale029-full-diversity-eeb38f5-20260711T1505Z/runs/gdn-full-diversity-hard-resume7000-s8000-20260711T2308Z-eeb38f5/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale029-full-diversity-eeb38f5-20260711T1505Z/runs/gdn-full-diversity-hard-resume7000-s8000-20260711T2308Z-eeb38f5/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.1289, blank_acc=0.6350, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.4473, valid=0.4473, solved=0.4473, blank_acc=0.7900, early=0.7908, late=0.7898, early_late_gap=+0.0010
- b56_64 (56-64, n=512): exact=0.1621, valid=0.1621, solved=0.1621, blank_acc=0.6322, early=0.6306, late=0.6280, early_late_gap=+0.0026

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale029-full-diversity-eeb38f5-20260711T1505Z/runs/gdn-full-diversity-hard-resume7000-s8000-20260711T2308Z-eeb38f5/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale029-full-diversity-eeb38f5-20260711T1505Z/runs/gdn-full-diversity-hard-resume7000-s8000-20260711T2308Z-eeb38f5/output/case_bank
