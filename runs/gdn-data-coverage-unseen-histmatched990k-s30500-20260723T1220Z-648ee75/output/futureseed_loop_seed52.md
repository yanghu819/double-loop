# 9x9 GDN-Triton FutureSeed Loop Study

Mainline mechanism: GDN-Triton recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.5620, total=0.6750, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9085, loop_last_loss=0.5620, sec=1904.7
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5967, early=0.5916, late=0.5933, early_late_gap=-0.0017
  future_seed: fs_gate=0.408, fs_update=1.000, fs_state_norm=13.041, fs_decay=0.00, fs_raw_rms=0.335, fs_raw_rms_std=0.035, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.1270, valid=0.1270, solved=0.1270, blank_acc=0.7007, early=0.7006, late=0.7008, early_late_gap=-0.0002
  future_seed: fs_gate=0.408, fs_update=1.000, fs_state_norm=13.041, fs_decay=0.00, fs_raw_rms=0.284, fs_raw_rms_std=0.031, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.3496, valid=0.3496, solved=0.3496, blank_acc=0.7478, early=0.7443, late=0.7462, early_late_gap=-0.0019
  future_seed: fs_gate=0.408, fs_update=1.000, fs_state_norm=13.041, fs_decay=0.00, fs_raw_rms=0.303, fs_raw_rms_std=0.049, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.4609, valid=0.4609, solved=0.4609, blank_acc=0.7608, early=0.7556, late=0.7625, early_late_gap=-0.0069
  future_seed: fs_gate=0.408, fs_update=1.000, fs_state_norm=13.037, fs_decay=0.00, fs_raw_rms=0.367, fs_raw_rms_std=0.078, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.4766, valid=0.4766, solved=0.4766, blank_acc=0.7655, early=0.7613, late=0.7648, early_late_gap=-0.0034
  future_seed: fs_gate=0.408, fs_update=1.000, fs_state_norm=13.028, fs_decay=0.00, fs_raw_rms=0.390, fs_raw_rms_std=0.096, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale036-matched-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-s30500-20260723T1220Z-648ee75/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale036-matched-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-s30500-20260723T1220Z-648ee75/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 1, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale036-matched-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-s30500-20260723T1220Z-648ee75/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale036-matched-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-s30500-20260723T1220Z-648ee75/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.6016, blank_acc=0.8371, selected={'solved_by_loop': 4, 'almost_solved': 2, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale036-matched-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-s30500-20260723T1220Z-648ee75/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale036-matched-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-s30500-20260723T1220Z-648ee75/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.3477, blank_acc=0.7068, selected={'solved_by_loop': 4, 'almost_solved': 2, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.6270, valid=0.6270, solved=0.6270, blank_acc=0.8447, early=0.8446, late=0.8418, early_late_gap=+0.0028
- b56_64 (56-64, n=512): exact=0.3711, valid=0.3711, solved=0.3711, blank_acc=0.7069, early=0.7046, late=0.7020, early_late_gap=+0.0026

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale036-matched-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-s30500-20260723T1220Z-648ee75/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale036-matched-648ee75-20260723/runs/gdn-data-coverage-unseen-histmatched990k-s30500-20260723T1220Z-648ee75/output/case_bank
