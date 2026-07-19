# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.1141, total=1.1183, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.1293, loop_last_loss=1.1141, sec=568.3
- loop 1: exact=0.0010, valid=0.0010, solved=0.0010, blank_acc=0.4626, early=0.4653, late=0.4706, early_late_gap=-0.0053
  future_seed: fs_gate=0.483, fs_update=1.000, fs_state_norm=7.722, fs_decay=0.00, fs_raw_rms=0.250, fs_raw_rms_std=0.019, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0010, valid=0.0010, solved=0.0010, blank_acc=0.4693, early=0.4742, late=0.4748, early_late_gap=-0.0007
  future_seed: fs_gate=0.483, fs_update=1.000, fs_state_norm=7.722, fs_decay=0.00, fs_raw_rms=0.246, fs_raw_rms_std=0.020, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0010, valid=0.0010, solved=0.0010, blank_acc=0.4696, early=0.4742, late=0.4749, early_late_gap=-0.0007
  future_seed: fs_gate=0.483, fs_update=1.000, fs_state_norm=7.722, fs_decay=0.00, fs_raw_rms=0.246, fs_raw_rms_std=0.020, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0010, valid=0.0010, solved=0.0010, blank_acc=0.4697, early=0.4742, late=0.4752, early_late_gap=-0.0010
  future_seed: fs_gate=0.483, fs_update=1.000, fs_state_norm=7.722, fs_decay=0.00, fs_raw_rms=0.245, fs_raw_rms_std=0.020, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/fla-kda-bf4b338-20260719/runs/fla-kda-futureseed-sudoku-gate-20260719T043622Z-bf4b338/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/fla-kda-bf4b338-20260719/runs/fla-kda-futureseed-sudoku-gate-20260719T043622Z-bf4b338/output/case_bank/official_b46_50/cases.json; final_loop=4, exact=0.0156, blank_acc=0.8960, selected={'solved_by_loop': 3, 'almost_solved': 4, 'hard_failure': 4}
- official_b51_55: index=/huyang2/double-loop/.worktrees/fla-kda-bf4b338-20260719/runs/fla-kda-futureseed-sudoku-gate-20260719T043622Z-bf4b338/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/fla-kda-bf4b338-20260719/runs/fla-kda-futureseed-sudoku-gate-20260719T043622Z-bf4b338/output/case_bank/official_b51_55/cases.json; final_loop=4, exact=0.0000, blank_acc=0.4879, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/fla-kda-bf4b338-20260719/runs/fla-kda-futureseed-sudoku-gate-20260719T043622Z-bf4b338/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/fla-kda-bf4b338-20260719/runs/fla-kda-futureseed-sudoku-gate-20260719T043622Z-bf4b338/output/case_bank/official_b56_64/cases.json; final_loop=4, exact=0.0000, blank_acc=0.4429, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=1024): exact=0.0127, valid=0.0127, solved=0.0127, blank_acc=0.8970, early=0.8910, late=0.9238, early_late_gap=-0.0328
- b51_55 (51-55, n=1024): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4922, early=0.4891, late=0.5041, early_late_gap=-0.0150
- b56_64 (56-64, n=1024): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4410, early=0.4426, late=0.4462, early_late_gap=-0.0036

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/fla-kda-bf4b338-20260719/runs/fla-kda-futureseed-sudoku-gate-20260719T043622Z-bf4b338/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/fla-kda-bf4b338-20260719/runs/fla-kda-futureseed-sudoku-gate-20260719T043622Z-bf4b338/output/case_bank
