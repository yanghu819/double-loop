# 9x9 KDA (official FLA) FutureSeed Loop Study

Mainline mechanism: KDA (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0051, total=1.0107, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0307, loop_last_loss=1.0051, sec=2690.5
- loop 1: exact=0.0117, valid=0.0117, solved=0.0117, blank_acc=0.4915, early=0.4940, late=0.4903, early_late_gap=+0.0037
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=22.598, fs_decay=0.00, fs_raw_rms=0.238, fs_raw_rms_std=0.011, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5035, early=0.5062, late=0.5023, early_late_gap=+0.0039
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=22.598, fs_decay=0.00, fs_raw_rms=0.230, fs_raw_rms_std=0.012, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5030, early=0.5040, late=0.5033, early_late_gap=+0.0007
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=22.598, fs_decay=0.00, fs_raw_rms=0.228, fs_raw_rms_std=0.012, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5033, early=0.5038, late=0.5040, early_late_gap=-0.0003
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=22.598, fs_decay=0.00, fs_raw_rms=0.227, fs_raw_rms_std=0.012, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5035, early=0.5044, late=0.5036, early_late_gap=+0.0008
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=22.598, fs_decay=0.00, fs_raw_rms=0.227, fs_raw_rms_std=0.012, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-kda-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-kda-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.7461, blank_acc=0.9907, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 2}
- official_b51_55: index=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-kda-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-kda-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5172, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-kda-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-kda-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4720, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.7285, valid=0.7285, solved=0.7285, blank_acc=0.9899, early=0.9882, late=0.9909, early_late_gap=-0.0027
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5257, early=0.5258, late=0.5215, early_late_gap=+0.0043
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4674, early=0.4662, late=0.4623, early_late_gap=+0.0038

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-kda-s500-clean-20260724T020321Z-3bc5dec/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-kda-s500-clean-20260724T020321Z-3bc5dec/output/case_bank
