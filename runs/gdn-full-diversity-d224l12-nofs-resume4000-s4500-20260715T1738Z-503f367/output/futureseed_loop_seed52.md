# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.6372, total=1.6376, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.6391, loop_last_loss=1.6372, sec=1450.5
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2745, early=0.2039, late=0.3481, early_late_gap=-0.1442
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2755, early=0.2030, late=0.3519, early_late_gap=-0.1489
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2756, early=0.2048, late=0.3495, early_late_gap=-0.1447
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2758, early=0.2048, late=0.3502, early_late_gap=-0.1455
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2760, early=0.2044, late=0.3501, early_late_gap=-0.1458
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume4000-s4500-20260715T1738Z-503f367/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume4000-s4500-20260715T1738Z-503f367/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4190, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume4000-s4500-20260715T1738Z-503f367/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume4000-s4500-20260715T1738Z-503f367/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.2913, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume4000-s4500-20260715T1738Z-503f367/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume4000-s4500-20260715T1738Z-503f367/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.2714, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4187, early=0.2631, late=0.5926, early_late_gap=-0.3295
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2862, early=0.2093, late=0.3719, early_late_gap=-0.1626
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2641, early=0.2033, late=0.3263, early_late_gap=-0.1230

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume4000-s4500-20260715T1738Z-503f367/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume4000-s4500-20260715T1738Z-503f367/output/case_bank
