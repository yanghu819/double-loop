# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.6054, total=0.7169, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9250, loop_last_loss=0.6054, sec=13759.8
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5892, early=0.5894, late=0.5878, early_late_gap=+0.0017
  future_seed: fs_gate=0.422, fs_update=1.000, fs_state_norm=13.516, fs_decay=0.00, fs_raw_rms=0.311, fs_raw_rms_std=0.031, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0840, valid=0.0840, solved=0.0840, blank_acc=0.6884, early=0.6874, late=0.6843, early_late_gap=+0.0031
  future_seed: fs_gate=0.422, fs_update=1.000, fs_state_norm=13.516, fs_decay=0.00, fs_raw_rms=0.276, fs_raw_rms_std=0.029, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.3242, valid=0.3242, solved=0.3242, blank_acc=0.7325, early=0.7308, late=0.7254, early_late_gap=+0.0054
  future_seed: fs_gate=0.422, fs_update=1.000, fs_state_norm=13.516, fs_decay=0.00, fs_raw_rms=0.298, fs_raw_rms_std=0.049, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.4160, valid=0.4160, solved=0.4160, blank_acc=0.7507, early=0.7471, late=0.7470, early_late_gap=+0.0001
  future_seed: fs_gate=0.422, fs_update=1.000, fs_state_norm=13.515, fs_decay=0.00, fs_raw_rms=0.354, fs_raw_rms_std=0.087, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.4375, valid=0.4375, solved=0.4375, blank_acc=0.7538, early=0.7511, late=0.7507, early_late_gap=+0.0004
  future_seed: fs_gate=0.422, fs_update=1.000, fs_state_norm=13.512, fs_decay=0.00, fs_raw_rms=0.374, fs_raw_rms_std=0.105, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale034-step20000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s20000-lease-20260721T110943Z-f8e009b/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-step20000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s20000-lease-20260721T110943Z-f8e009b/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 1, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale034-step20000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s20000-lease-20260721T110943Z-f8e009b/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-step20000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s20000-lease-20260721T110943Z-f8e009b/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.5664, blank_acc=0.8336, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale034-step20000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s20000-lease-20260721T110943Z-f8e009b/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-step20000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s20000-lease-20260721T110943Z-f8e009b/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.2969, blank_acc=0.6896, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.5840, valid=0.5840, solved=0.5840, blank_acc=0.8351, early=0.8363, late=0.8360, early_late_gap=+0.0002
- b56_64 (56-64, n=512): exact=0.3125, valid=0.3125, solved=0.3125, blank_acc=0.6942, early=0.6903, late=0.6880, early_late_gap=+0.0023

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale034-step20000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s20000-lease-20260721T110943Z-f8e009b/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale034-step20000-f8e009b-20260721/runs/gdn-full-diversity-d224l12-upperbound-s20000-lease-20260721T110943Z-f8e009b/output/case_bank
