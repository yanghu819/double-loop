# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.5485, total=0.6582, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.8803, loop_last_loss=0.5485, sec=6946.0
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5921, early=0.5878, late=0.5909, early_late_gap=-0.0031
  future_seed: fs_gate=0.408, fs_update=1.000, fs_state_norm=13.051, fs_decay=0.00, fs_raw_rms=0.328, fs_raw_rms_std=0.033, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0898, valid=0.0898, solved=0.0898, blank_acc=0.6959, early=0.6878, late=0.6967, early_late_gap=-0.0088
  future_seed: fs_gate=0.408, fs_update=1.000, fs_state_norm=13.051, fs_decay=0.00, fs_raw_rms=0.279, fs_raw_rms_std=0.030, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.3379, valid=0.3379, solved=0.3379, blank_acc=0.7413, early=0.7356, late=0.7445, early_late_gap=-0.0089
  future_seed: fs_gate=0.408, fs_update=1.000, fs_state_norm=13.051, fs_decay=0.00, fs_raw_rms=0.309, fs_raw_rms_std=0.056, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.4570, valid=0.4570, solved=0.4570, blank_acc=0.7598, early=0.7563, late=0.7596, early_late_gap=-0.0033
  future_seed: fs_gate=0.408, fs_update=1.000, fs_state_norm=13.045, fs_decay=0.00, fs_raw_rms=0.392, fs_raw_rms_std=0.099, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.4805, valid=0.4805, solved=0.4805, blank_acc=0.7655, early=0.7615, late=0.7658, early_late_gap=-0.0044
  future_seed: fs_gate=0.408, fs_update=1.000, fs_state_norm=13.033, fs_decay=0.00, fs_raw_rms=0.419, fs_raw_rms_std=0.122, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale034-step30000-f8e009b-20260722/runs/gdn-full-diversity-d224l12-upperbound-s30000-lease-20260722T084720Z-f8e009b/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-step30000-f8e009b-20260722/runs/gdn-full-diversity-d224l12-upperbound-s30000-lease-20260722T084720Z-f8e009b/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale034-step30000-f8e009b-20260722/runs/gdn-full-diversity-d224l12-upperbound-s30000-lease-20260722T084720Z-f8e009b/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-step30000-f8e009b-20260722/runs/gdn-full-diversity-d224l12-upperbound-s30000-lease-20260722T084720Z-f8e009b/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.6289, blank_acc=0.8466, selected={'solved_by_loop': 4, 'almost_solved': 1, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale034-step30000-f8e009b-20260722/runs/gdn-full-diversity-d224l12-upperbound-s30000-lease-20260722T084720Z-f8e009b/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-step30000-f8e009b-20260722/runs/gdn-full-diversity-d224l12-upperbound-s30000-lease-20260722T084720Z-f8e009b/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.3516, blank_acc=0.7033, selected={'solved_by_loop': 4, 'almost_solved': 2, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.6152, valid=0.6152, solved=0.6152, blank_acc=0.8472, early=0.8476, late=0.8429, early_late_gap=+0.0048
- b56_64 (56-64, n=512): exact=0.3848, valid=0.3848, solved=0.3848, blank_acc=0.7114, early=0.7062, late=0.7047, early_late_gap=+0.0016

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale034-step30000-f8e009b-20260722/runs/gdn-full-diversity-d224l12-upperbound-s30000-lease-20260722T084720Z-f8e009b/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale034-step30000-f8e009b-20260722/runs/gdn-full-diversity-d224l12-upperbound-s30000-lease-20260722T084720Z-f8e009b/output/case_bank
