# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0432, total=1.0508, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0786, loop_last_loss=1.0432, sec=1718.6
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4681, early=0.4661, late=0.4709, early_late_gap=-0.0048
  future_seed: fs_gate=0.527, fs_update=1.000, fs_state_norm=16.878, fs_decay=0.00, fs_raw_rms=0.861, fs_raw_rms_std=0.073, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0059, valid=0.0059, solved=0.0059, blank_acc=0.4842, early=0.4816, late=0.4839, early_late_gap=-0.0023
  future_seed: fs_gate=0.527, fs_update=1.000, fs_state_norm=16.878, fs_decay=0.00, fs_raw_rms=0.837, fs_raw_rms_std=0.078, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.4859, early=0.4819, late=0.4862, early_late_gap=-0.0043
  future_seed: fs_gate=0.527, fs_update=1.000, fs_state_norm=16.878, fs_decay=0.00, fs_raw_rms=0.831, fs_raw_rms_std=0.079, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.4866, early=0.4821, late=0.4868, early_late_gap=-0.0047
  future_seed: fs_gate=0.527, fs_update=1.000, fs_state_norm=16.878, fs_decay=0.00, fs_raw_rms=0.830, fs_raw_rms_std=0.079, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.4867, early=0.4827, late=0.4864, early_late_gap=-0.0037
  future_seed: fs_gate=0.527, fs_update=1.000, fs_state_norm=16.878, fs_decay=0.00, fs_raw_rms=0.829, fs_raw_rms_std=0.079, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-rwkv-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-rwkv-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.3398, blank_acc=0.9658, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b51_55: index=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-rwkv-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-rwkv-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5000, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-rwkv-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-rwkv-s500-clean-20260724T020321Z-3bc5dec/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4555, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.3359, valid=0.3359, solved=0.3359, blank_acc=0.9659, early=0.9620, late=0.9715, early_late_gap=-0.0094
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5097, early=0.5061, late=0.5110, early_late_gap=-0.0049
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4553, early=0.4490, late=0.4560, early_late_gap=-0.0071

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-rwkv-s500-clean-20260724T020321Z-3bc5dec/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/artifacts/worktrees/baseline-clean-20260724-3bc5dec/runs/sudoku-backbone-rwkv-s500-clean-20260724T020321Z-3bc5dec/output/case_bank
