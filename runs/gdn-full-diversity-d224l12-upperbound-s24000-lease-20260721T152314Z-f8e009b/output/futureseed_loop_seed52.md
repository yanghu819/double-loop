# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.5579, total=0.6591, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.8891, loop_last_loss=0.5579, sec=13791.6
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5932, early=0.5923, late=0.5915, early_late_gap=+0.0008
  future_seed: fs_gate=0.416, fs_update=1.000, fs_state_norm=13.322, fs_decay=0.00, fs_raw_rms=0.313, fs_raw_rms_std=0.031, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.1055, valid=0.1055, solved=0.1055, blank_acc=0.6983, early=0.6924, late=0.6992, early_late_gap=-0.0068
  future_seed: fs_gate=0.416, fs_update=1.000, fs_state_norm=13.322, fs_decay=0.00, fs_raw_rms=0.274, fs_raw_rms_std=0.030, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.3477, valid=0.3477, solved=0.3477, blank_acc=0.7443, early=0.7401, late=0.7459, early_late_gap=-0.0058
  future_seed: fs_gate=0.416, fs_update=1.000, fs_state_norm=13.321, fs_decay=0.00, fs_raw_rms=0.299, fs_raw_rms_std=0.055, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.4414, valid=0.4414, solved=0.4414, blank_acc=0.7606, early=0.7565, late=0.7619, early_late_gap=-0.0054
  future_seed: fs_gate=0.416, fs_update=1.000, fs_state_norm=13.308, fs_decay=0.00, fs_raw_rms=0.365, fs_raw_rms_std=0.096, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.4648, valid=0.4648, solved=0.4648, blank_acc=0.7643, early=0.7616, late=0.7639, early_late_gap=-0.0024
  future_seed: fs_gate=0.416, fs_update=1.000, fs_state_norm=13.298, fs_decay=0.00, fs_raw_rms=0.390, fs_raw_rms_std=0.118, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale034-step24000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s24000-lease-20260721T152314Z-f8e009b/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-step24000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s24000-lease-20260721T152314Z-f8e009b/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 1, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale034-step24000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s24000-lease-20260721T152314Z-f8e009b/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-step24000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s24000-lease-20260721T152314Z-f8e009b/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.6055, blank_acc=0.8419, selected={'solved_by_loop': 4, 'almost_solved': 3, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale034-step24000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s24000-lease-20260721T152314Z-f8e009b/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-step24000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s24000-lease-20260721T152314Z-f8e009b/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.3008, blank_acc=0.6948, selected={'solved_by_loop': 4, 'almost_solved': 3, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.6387, valid=0.6387, solved=0.6387, blank_acc=0.8513, early=0.8490, late=0.8491, early_late_gap=-0.0001
- b56_64 (56-64, n=512): exact=0.2949, valid=0.2949, solved=0.2949, blank_acc=0.6826, early=0.6805, late=0.6754, early_late_gap=+0.0051

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale034-step24000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s24000-lease-20260721T152314Z-f8e009b/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale034-step24000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s24000-lease-20260721T152314Z-f8e009b/output/case_bank
