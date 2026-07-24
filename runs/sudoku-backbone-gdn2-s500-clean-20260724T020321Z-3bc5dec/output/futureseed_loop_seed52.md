# 9x9 GDN2 (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN2 (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0078, total=1.0145, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0371, loop_last_loss=1.0078, sec=2738.6
- loop 1: exact=0.0117, valid=0.0117, solved=0.0117, blank_acc=0.4873, early=0.4836, late=0.4892, early_late_gap=-0.0057
  future_seed: fs_gate=0.503, fs_update=1.000, fs_state_norm=22.781, fs_decay=0.00, fs_raw_rms=0.495, fs_raw_rms_std=0.035, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5024, early=0.4987, late=0.5070, early_late_gap=-0.0082
  future_seed: fs_gate=0.503, fs_update=1.000, fs_state_norm=22.781, fs_decay=0.00, fs_raw_rms=0.476, fs_raw_rms_std=0.038, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5017, early=0.4967, late=0.5065, early_late_gap=-0.0098
  future_seed: fs_gate=0.503, fs_update=1.000, fs_state_norm=22.781, fs_decay=0.00, fs_raw_rms=0.468, fs_raw_rms_std=0.038, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5019, early=0.4957, late=0.5069, early_late_gap=-0.0112
  future_seed: fs_gate=0.503, fs_update=1.000, fs_state_norm=22.781, fs_decay=0.00, fs_raw_rms=0.466, fs_raw_rms_std=0.038, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5016, early=0.4954, late=0.5069, early_late_gap=-0.0115
  future_seed: fs_gate=0.503, fs_update=1.000, fs_state_norm=22.781, fs_decay=0.00, fs_raw_rms=0.465, fs_raw_rms_std=0.038, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn2-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn2-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.7383, blank_acc=0.9892, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 2}
- official_b51_55: index=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn2-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn2-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5234, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn2-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn2-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4665, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.7676, valid=0.7676, solved=0.7676, blank_acc=0.9905, early=0.9875, late=0.9916, early_late_gap=-0.0041
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5205, early=0.5166, late=0.5217, early_late_gap=-0.0050
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4689, early=0.4649, late=0.4706, early_late_gap=-0.0057

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn2-s500-clean-20260724T020321Z-3bc5dec/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn2-s500-clean-20260724T020321Z-3bc5dec/output/case_bank
