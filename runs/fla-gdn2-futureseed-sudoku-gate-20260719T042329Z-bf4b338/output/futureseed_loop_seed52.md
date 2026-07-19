# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0997, total=1.1050, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.1194, loop_last_loss=1.0997, sec=669.5
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4715, early=0.4686, late=0.4749, early_late_gap=-0.0063
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=7.964, fs_decay=0.00, fs_raw_rms=0.518, fs_raw_rms_std=0.051, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0010, valid=0.0010, solved=0.0010, blank_acc=0.4814, early=0.4795, late=0.4850, early_late_gap=-0.0056
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=7.964, fs_decay=0.00, fs_raw_rms=0.510, fs_raw_rms_std=0.052, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0020, valid=0.0020, solved=0.0020, blank_acc=0.4815, early=0.4797, late=0.4843, early_late_gap=-0.0046
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=7.964, fs_decay=0.00, fs_raw_rms=0.507, fs_raw_rms_std=0.052, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0020, valid=0.0020, solved=0.0020, blank_acc=0.4813, early=0.4803, late=0.4837, early_late_gap=-0.0033
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=7.964, fs_decay=0.00, fs_raw_rms=0.507, fs_raw_rms_std=0.052, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/fla-gdn2-bf4b338-20260719/runs/fla-gdn2-futureseed-sudoku-gate-20260719T042329Z-bf4b338/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/fla-gdn2-bf4b338-20260719/runs/fla-gdn2-futureseed-sudoku-gate-20260719T042329Z-bf4b338/output/case_bank/official_b46_50/cases.json; final_loop=4, exact=0.0547, blank_acc=0.9278, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b51_55: index=/huyang2/double-loop/.worktrees/fla-gdn2-bf4b338-20260719/runs/fla-gdn2-futureseed-sudoku-gate-20260719T042329Z-bf4b338/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/fla-gdn2-bf4b338-20260719/runs/fla-gdn2-futureseed-sudoku-gate-20260719T042329Z-bf4b338/output/case_bank/official_b51_55/cases.json; final_loop=4, exact=0.0000, blank_acc=0.4984, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/fla-gdn2-bf4b338-20260719/runs/fla-gdn2-futureseed-sudoku-gate-20260719T042329Z-bf4b338/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/fla-gdn2-bf4b338-20260719/runs/fla-gdn2-futureseed-sudoku-gate-20260719T042329Z-bf4b338/output/case_bank/official_b56_64/cases.json; final_loop=4, exact=0.0000, blank_acc=0.4485, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=1024): exact=0.0732, valid=0.0732, solved=0.0732, blank_acc=0.9271, early=0.9201, late=0.9378, early_late_gap=-0.0177
- b51_55 (51-55, n=1024): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5010, early=0.4964, late=0.5023, early_late_gap=-0.0059
- b56_64 (56-64, n=1024): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4513, early=0.4528, late=0.4470, early_late_gap=+0.0058

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/fla-gdn2-bf4b338-20260719/runs/fla-gdn2-futureseed-sudoku-gate-20260719T042329Z-bf4b338/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/fla-gdn2-bf4b338-20260719/runs/fla-gdn2-futureseed-sudoku-gate-20260719T042329Z-bf4b338/output/case_bank
