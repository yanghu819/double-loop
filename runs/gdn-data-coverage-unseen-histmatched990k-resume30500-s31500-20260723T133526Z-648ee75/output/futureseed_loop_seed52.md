# 9x9 GDN-Triton FutureSeed Loop Study

Mainline mechanism: GDN-Triton recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.5197, total=0.6427, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.8917, loop_last_loss=0.5197, sec=3495.6
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5962, early=0.5928, late=0.5943, early_late_gap=-0.0015
  future_seed: fs_gate=0.407, fs_update=1.000, fs_state_norm=13.011, fs_decay=0.00, fs_raw_rms=0.332, fs_raw_rms_std=0.034, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.1289, valid=0.1289, solved=0.1289, blank_acc=0.6931, early=0.6889, late=0.6902, early_late_gap=-0.0013
  future_seed: fs_gate=0.407, fs_update=1.000, fs_state_norm=13.011, fs_decay=0.00, fs_raw_rms=0.286, fs_raw_rms_std=0.031, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.3496, valid=0.3496, solved=0.3496, blank_acc=0.7283, early=0.7236, late=0.7240, early_late_gap=-0.0004
  future_seed: fs_gate=0.407, fs_update=1.000, fs_state_norm=13.010, fs_decay=0.00, fs_raw_rms=0.315, fs_raw_rms_std=0.054, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.4355, valid=0.4355, solved=0.4355, blank_acc=0.7484, early=0.7469, late=0.7471, early_late_gap=-0.0002
  future_seed: fs_gate=0.407, fs_update=1.000, fs_state_norm=12.999, fs_decay=0.00, fs_raw_rms=0.382, fs_raw_rms_std=0.090, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.4551, valid=0.4551, solved=0.4551, blank_acc=0.7528, early=0.7504, late=0.7493, early_late_gap=+0.0010
  future_seed: fs_gate=0.407, fs_update=1.000, fs_state_norm=12.992, fs_decay=0.00, fs_raw_rms=0.400, fs_raw_rms_std=0.104, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale036-matched-resume-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-resume30500-s31500-20260723T133526Z-648ee75/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale036-matched-resume-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-resume30500-s31500-20260723T133526Z-648ee75/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale036-matched-resume-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-resume30500-s31500-20260723T133526Z-648ee75/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale036-matched-resume-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-resume30500-s31500-20260723T133526Z-648ee75/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.6406, blank_acc=0.8509, selected={'solved_by_loop': 4, 'almost_solved': 2, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale036-matched-resume-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-resume30500-s31500-20260723T133526Z-648ee75/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale036-matched-resume-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-resume30500-s31500-20260723T133526Z-648ee75/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.3789, blank_acc=0.7126, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.6367, valid=0.6367, solved=0.6367, blank_acc=0.8500, early=0.8520, late=0.8465, early_late_gap=+0.0055
- b56_64 (56-64, n=512): exact=0.3711, valid=0.3711, solved=0.3711, blank_acc=0.7038, early=0.6987, late=0.7028, early_late_gap=-0.0042

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale036-matched-resume-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-resume30500-s31500-20260723T133526Z-648ee75/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale036-matched-resume-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-resume30500-s31500-20260723T133526Z-648ee75/output/case_bank
