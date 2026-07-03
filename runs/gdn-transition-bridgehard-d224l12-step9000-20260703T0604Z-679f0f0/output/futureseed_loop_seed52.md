# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.2735, total=0.4980, loop_loss=all, noise=feature_diff, buffer=8192, dtype=float32, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0432, loop_last_loss=0.2735, sec=4752.6
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5381, early=0.5373, late=0.5357, early_late_gap=+0.0016
  future_seed: fs_gate=0.447, fs_update=1.000, fs_state_norm=14.314, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0332, valid=0.0332, solved=0.0332, blank_acc=0.5905, early=0.5869, late=0.5994, early_late_gap=-0.0125
  future_seed: fs_gate=0.447, fs_update=1.000, fs_state_norm=14.314, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1523, valid=0.1523, solved=0.1523, blank_acc=0.6114, early=0.6117, late=0.6117, early_late_gap=+0.0000
  future_seed: fs_gate=0.447, fs_update=1.000, fs_state_norm=14.314, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.1934, valid=0.1934, solved=0.1934, blank_acc=0.6207, early=0.6193, late=0.6210, early_late_gap=-0.0017
  future_seed: fs_gate=0.447, fs_update=1.000, fs_state_norm=14.314, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.1973, valid=0.1973, solved=0.1973, blank_acc=0.6231, early=0.6223, late=0.6235, early_late_gap=-0.0013
  future_seed: fs_gate=0.447, fs_update=1.000, fs_state_norm=14.314, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/runs/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/runs/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 4, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/runs/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/runs/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.3281, blank_acc=0.7176, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/runs/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/runs/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.1406, blank_acc=0.5732, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.2871, valid=0.2891, solved=0.2871, blank_acc=0.6942, early=0.6961, late=0.6914, early_late_gap=+0.0047
- b56_64 (56-64, n=512): exact=0.1250, valid=0.1250, solved=0.1250, blank_acc=0.5624, early=0.5631, late=0.5586, early_late_gap=+0.0045

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/runs/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/runs/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0/output/case_bank
