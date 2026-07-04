# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0000, total=0.1645, loop_loss=all, noise=feature_diff, buffer=8192, dtype=float32, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.7632, loop_last_loss=0.0000, sec=10226.7
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5381, early=0.5370, late=0.5399, early_late_gap=-0.0029
  future_seed: fs_gate=0.437, fs_update=1.000, fs_state_norm=13.990, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0547, valid=0.0547, solved=0.0547, blank_acc=0.5912, early=0.5878, late=0.5916, early_late_gap=-0.0038
  future_seed: fs_gate=0.437, fs_update=1.000, fs_state_norm=13.990, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1895, valid=0.1895, solved=0.1895, blank_acc=0.6180, early=0.6121, late=0.6228, early_late_gap=-0.0107
  future_seed: fs_gate=0.437, fs_update=1.000, fs_state_norm=13.990, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.2266, valid=0.2285, solved=0.2266, blank_acc=0.6256, early=0.6165, late=0.6323, early_late_gap=-0.0157
  future_seed: fs_gate=0.437, fs_update=1.000, fs_state_norm=13.990, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.2363, valid=0.2402, solved=0.2363, blank_acc=0.6287, early=0.6216, late=0.6348, early_late_gap=-0.0131
  future_seed: fs_gate=0.437, fs_update=1.000, fs_state_norm=13.990, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 4, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.3711, blank_acc=0.7287, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.1211, blank_acc=0.5644, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.3555, valid=0.3574, solved=0.3555, blank_acc=0.7312, early=0.7342, late=0.7331, early_late_gap=+0.0012
- b56_64 (56-64, n=512): exact=0.1582, valid=0.1582, solved=0.1582, blank_acc=0.5647, early=0.5556, late=0.5610, early_late_gap=-0.0054

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/output/case_bank
