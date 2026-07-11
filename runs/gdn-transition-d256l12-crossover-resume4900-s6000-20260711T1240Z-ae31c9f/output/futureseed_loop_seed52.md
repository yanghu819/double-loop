# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.7586, total=0.8255, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9923, loop_last_loss=0.7586, sec=3997.4
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5383, early=0.5350, late=0.5419, early_late_gap=-0.0069
  future_seed: fs_gate=0.471, fs_update=1.000, fs_state_norm=15.059, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0273, valid=0.0273, solved=0.0273, blank_acc=0.6021, early=0.6008, late=0.6036, early_late_gap=-0.0028
  future_seed: fs_gate=0.471, fs_update=1.000, fs_state_norm=15.059, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0938, valid=0.0938, solved=0.0938, blank_acc=0.6272, early=0.6206, late=0.6290, early_late_gap=-0.0084
  future_seed: fs_gate=0.471, fs_update=1.000, fs_state_norm=15.059, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.1328, valid=0.1328, solved=0.1328, blank_acc=0.6323, early=0.6269, late=0.6319, early_late_gap=-0.0051
  future_seed: fs_gate=0.471, fs_update=1.000, fs_state_norm=15.059, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.1367, valid=0.1367, solved=0.1367, blank_acc=0.6327, early=0.6270, late=0.6324, early_late_gap=-0.0054
  future_seed: fs_gate=0.471, fs_update=1.000, fs_state_norm=15.059, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z/runs/gdn-transition-d256l12-crossover-resume4900-s6000-20260711T1240Z-ae31c9f/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z/runs/gdn-transition-d256l12-crossover-resume4900-s6000-20260711T1240Z-ae31c9f/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 4, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z/runs/gdn-transition-d256l12-crossover-resume4900-s6000-20260711T1240Z-ae31c9f/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z/runs/gdn-transition-d256l12-crossover-resume4900-s6000-20260711T1240Z-ae31c9f/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.1016, blank_acc=0.6886, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z/runs/gdn-transition-d256l12-crossover-resume4900-s6000-20260711T1240Z-ae31c9f/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z/runs/gdn-transition-d256l12-crossover-resume4900-s6000-20260711T1240Z-ae31c9f/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0586, blank_acc=0.5809, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.1504, valid=0.1504, solved=0.1504, blank_acc=0.6956, early=0.6888, late=0.6954, early_late_gap=-0.0065
- b56_64 (56-64, n=512): exact=0.0781, valid=0.0781, solved=0.0781, blank_acc=0.5834, early=0.5776, late=0.5825, early_late_gap=-0.0049

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z/runs/gdn-transition-d256l12-crossover-resume4900-s6000-20260711T1240Z-ae31c9f/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z/runs/gdn-transition-d256l12-crossover-resume4900-s6000-20260711T1240Z-ae31c9f/output/case_bank
