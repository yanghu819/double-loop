# 9x9 FLA GDN FutureSeed Loop Study

Mainline mechanism: FLA GDN recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9413, total=0.9521, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9906, loop_last_loss=0.9413, sec=2573.1
- loop 1: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5070, early=0.5058, late=0.5071, early_late_gap=-0.0013
  future_seed: fs_gate=0.487, fs_update=1.000, fs_state_norm=22.029, fs_decay=0.00, fs_raw_rms=0.169, fs_raw_rms_std=0.009, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5287, early=0.5231, late=0.5298, early_late_gap=-0.0067
  future_seed: fs_gate=0.487, fs_update=1.000, fs_state_norm=22.029, fs_decay=0.00, fs_raw_rms=0.161, fs_raw_rms_std=0.010, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5313, early=0.5252, late=0.5358, early_late_gap=-0.0106
  future_seed: fs_gate=0.487, fs_update=1.000, fs_state_norm=22.029, fs_decay=0.00, fs_raw_rms=0.159, fs_raw_rms_std=0.010, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5313, early=0.5245, late=0.5353, early_late_gap=-0.0108
  future_seed: fs_gate=0.487, fs_update=1.000, fs_state_norm=22.029, fs_decay=0.00, fs_raw_rms=0.159, fs_raw_rms_std=0.010, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5318, early=0.5256, late=0.5356, early_late_gap=-0.0100
  future_seed: fs_gate=0.487, fs_update=1.000, fs_state_norm=22.029, fs_decay=0.00, fs_raw_rms=0.158, fs_raw_rms_std=0.010, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/fla-seed-survival-d9b4963-20260721/runs/fla-official-gdn-fs-d192l10-resume500-s1000-20260721T010000Z-87e66cf/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/fla-seed-survival-d9b4963-20260721/runs/fla-official-gdn-fs-d192l10-resume500-s1000-20260721T010000Z-87e66cf/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.9922, blank_acc=0.9996, selected={'solved_by_loop': 4, 'almost_solved': 2, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/fla-seed-survival-d9b4963-20260721/runs/fla-official-gdn-fs-d192l10-resume500-s1000-20260721T010000Z-87e66cf/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/fla-seed-survival-d9b4963-20260721/runs/fla-official-gdn-fs-d192l10-resume500-s1000-20260721T010000Z-87e66cf/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5440, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/fla-seed-survival-d9b4963-20260721/runs/fla-official-gdn-fs-d192l10-resume500-s1000-20260721T010000Z-87e66cf/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/fla-seed-survival-d9b4963-20260721/runs/fla-official-gdn-fs-d192l10-resume500-s1000-20260721T010000Z-87e66cf/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4911, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.9980, valid=0.9980, solved=0.9980, blank_acc=0.9998, early=0.9998, late=1.0000, early_late_gap=-0.0002
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5537, early=0.5534, late=0.5508, early_late_gap=+0.0026
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4909, early=0.4875, late=0.4842, early_late_gap=+0.0033

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/fla-seed-survival-d9b4963-20260721/runs/fla-official-gdn-fs-d192l10-resume500-s1000-20260721T010000Z-87e66cf/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/fla-seed-survival-d9b4963-20260721/runs/fla-official-gdn-fs-d192l10-resume500-s1000-20260721T010000Z-87e66cf/output/case_bank
