# 9x9 GDN (official FLA) FutureSeed Loop Study

Mainline mechanism: GDN (official FLA) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9964, total=1.0037, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0303, loop_last_loss=0.9964, sec=2768.0
- loop 1: exact=0.0117, valid=0.0117, solved=0.0117, blank_acc=0.4923, early=0.4882, late=0.4956, early_late_gap=-0.0075
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=22.096, fs_decay=0.00, fs_raw_rms=0.135, fs_raw_rms_std=0.008, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5063, early=0.5015, late=0.5096, early_late_gap=-0.0081
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=22.096, fs_decay=0.00, fs_raw_rms=0.134, fs_raw_rms_std=0.008, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5066, early=0.5008, late=0.5124, early_late_gap=-0.0116
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=22.096, fs_decay=0.00, fs_raw_rms=0.133, fs_raw_rms_std=0.008, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5061, early=0.4999, late=0.5121, early_late_gap=-0.0122
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=22.096, fs_decay=0.00, fs_raw_rms=0.133, fs_raw_rms_std=0.008, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5059, early=0.4998, late=0.5118, early_late_gap=-0.0120
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=22.096, fs_decay=0.00, fs_raw_rms=0.133, fs_raw_rms_std=0.008, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.8750, blank_acc=0.9954, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 1}
- official_b51_55: index=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5313, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4717, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.8652, valid=0.8652, solved=0.8652, blank_acc=0.9955, early=0.9940, late=0.9952, early_late_gap=-0.0011
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5279, early=0.5254, late=0.5256, early_late_gap=-0.0002
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4727, early=0.4686, late=0.4742, early_late_gap=-0.0056

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn-s500-clean-20260724T020321Z-3bc5dec/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-gdn-s500-clean-20260724T020321Z-3bc5dec/output/case_bank
