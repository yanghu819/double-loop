# 9x9 GDN FutureSeed Loop Study

Mainline mechanism: GDN recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.4485, total=0.5706, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.8564, loop_last_loss=0.4485, sec=1916.2
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5939, early=0.5883, late=0.5949, early_late_gap=-0.0066
  future_seed: fs_gate=0.407, fs_update=1.000, fs_state_norm=18.432, fs_decay=0.00, fs_raw_rms=0.358, fs_raw_rms_std=0.037, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.1133, valid=0.1133, solved=0.1133, blank_acc=0.6914, early=0.6902, late=0.6904, early_late_gap=-0.0001
  future_seed: fs_gate=0.407, fs_update=1.000, fs_state_norm=18.432, fs_decay=0.00, fs_raw_rms=0.308, fs_raw_rms_std=0.035, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.3359, valid=0.3359, solved=0.3359, blank_acc=0.7340, early=0.7344, late=0.7288, early_late_gap=+0.0056
  future_seed: fs_gate=0.407, fs_update=1.000, fs_state_norm=18.429, fs_decay=0.00, fs_raw_rms=0.328, fs_raw_rms_std=0.062, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.4336, valid=0.4336, solved=0.4336, blank_acc=0.7497, early=0.7443, late=0.7494, early_late_gap=-0.0051
  future_seed: fs_gate=0.407, fs_update=1.000, fs_state_norm=18.416, fs_decay=0.00, fs_raw_rms=0.401, fs_raw_rms_std=0.105, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.4551, valid=0.4551, solved=0.4551, blank_acc=0.7543, early=0.7506, late=0.7563, early_late_gap=-0.0057
  future_seed: fs_gate=0.407, fs_update=1.000, fs_state_norm=18.400, fs_decay=0.00, fs_raw_rms=0.425, fs_raw_rms_std=0.126, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale035-state-expandv8-83e7841-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s30500-resume-20260722T125827Z-83e7841/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale035-state-expandv8-83e7841-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s30500-resume-20260722T125827Z-83e7841/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.9961, blank_acc=1.0000, selected={'solved_by_loop': 1, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale035-state-expandv8-83e7841-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s30500-resume-20260722T125827Z-83e7841/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale035-state-expandv8-83e7841-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s30500-resume-20260722T125827Z-83e7841/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.6289, blank_acc=0.8466, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale035-state-expandv8-83e7841-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s30500-resume-20260722T125827Z-83e7841/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale035-state-expandv8-83e7841-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s30500-resume-20260722T125827Z-83e7841/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.3242, blank_acc=0.6891, selected={'solved_by_loop': 4, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=1.0000, early=1.0000, late=0.9999, early_late_gap=+0.0001
- b51_55 (51-55, n=512): exact=0.6172, valid=0.6172, solved=0.6172, blank_acc=0.8380, early=0.8395, late=0.8369, early_late_gap=+0.0026
- b56_64 (56-64, n=512): exact=0.3535, valid=0.3535, solved=0.3535, blank_acc=0.6991, early=0.6953, late=0.6960, early_late_gap=-0.0007

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale035-state-expandv8-83e7841-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s30500-resume-20260722T125827Z-83e7841/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale035-state-expandv8-83e7841-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s30500-resume-20260722T125827Z-83e7841/output/case_bank
