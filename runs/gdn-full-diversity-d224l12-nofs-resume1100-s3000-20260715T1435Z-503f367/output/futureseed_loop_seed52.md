# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.6379, total=1.6381, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.6391, loop_last_loss=1.6379, sec=5557.8
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2716, early=0.2013, late=0.3452, early_late_gap=-0.1439
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2723, early=0.2023, late=0.3466, early_late_gap=-0.1443
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2719, early=0.2027, late=0.3454, early_late_gap=-0.1427
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2714, early=0.2018, late=0.3449, early_late_gap=-0.1430
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2713, early=0.2015, late=0.3452, early_late_gap=-0.1437
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume1100-s3000-20260715T1435Z-503f367/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume1100-s3000-20260715T1435Z-503f367/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4206, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume1100-s3000-20260715T1435Z-503f367/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume1100-s3000-20260715T1435Z-503f367/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.2815, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume1100-s3000-20260715T1435Z-503f367/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume1100-s3000-20260715T1435Z-503f367/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.2685, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4125, early=0.2612, late=0.5797, early_late_gap=-0.3184
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2812, early=0.2042, late=0.3633, early_late_gap=-0.1591
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2623, early=0.2036, late=0.3219, early_late_gap=-0.1184

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume1100-s3000-20260715T1435Z-503f367/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale033-nofs-d224-503f367-20260715/runs/gdn-full-diversity-d224l12-nofs-resume1100-s3000-20260715T1435Z-503f367/output/case_bank
