# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.7943, total=0.8400, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9681, loop_last_loss=0.7943, sec=8645.1
- loop 1: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5215, early=0.5165, late=0.5234, early_late_gap=-0.0069
  future_seed: fs_gate=0.481, fs_update=1.000, fs_state_norm=15.378, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5675, early=0.5664, late=0.5699, early_late_gap=-0.0035
  future_seed: fs_gate=0.481, fs_update=1.000, fs_state_norm=15.378, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0371, valid=0.0371, solved=0.0371, blank_acc=0.5844, early=0.5836, late=0.5880, early_late_gap=-0.0044
  future_seed: fs_gate=0.481, fs_update=1.000, fs_state_norm=15.378, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0352, valid=0.0352, solved=0.0352, blank_acc=0.5882, early=0.5883, late=0.5921, early_late_gap=-0.0038
  future_seed: fs_gate=0.481, fs_update=1.000, fs_state_norm=15.378, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0410, valid=0.0410, solved=0.0410, blank_acc=0.5897, early=0.5904, late=0.5932, early_late_gap=-0.0029
  future_seed: fs_gate=0.481, fs_update=1.000, fs_state_norm=15.378, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-long-4346594-20260701T1125Z/runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-long-4346594-20260701T1125Z/runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.9948, blank_acc=0.9997, selected={'solved_by_loop': 3, 'almost_solved': 1, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-long-4346594-20260701T1125Z/runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-long-4346594-20260701T1125Z/runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0260, blank_acc=0.6522, selected={'solved_by_loop': 3, 'almost_solved': 3, 'hard_failure': 3}
- official_b56_64: index=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-long-4346594-20260701T1125Z/runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-long-4346594-20260701T1125Z/runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5231, selected={'solved_by_loop': 0, 'almost_solved': 3, 'hard_failure': 3}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.6441, early=0.6492, late=0.6387, early_late_gap=+0.0106
- b56_64 (56-64, n=512): exact=0.0137, valid=0.0137, solved=0.0137, blank_acc=0.5266, early=0.5226, late=0.5269, early_late_gap=-0.0043

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-long-4346594-20260701T1125Z/runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-long-4346594-20260701T1125Z/runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/output/case_bank
