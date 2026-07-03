# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.5736, total=0.6898, loop_loss=all, noise=feature_diff, buffer=8192, dtype=float32, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9704, loop_last_loss=0.5736, sec=2237.4
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5427, early=0.5337, late=0.5422, early_late_gap=-0.0086
  future_seed: fs_gate=0.458, fs_update=1.000, fs_state_norm=14.669, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.6080, early=0.6099, late=0.6068, early_late_gap=+0.0032
  future_seed: fs_gate=0.458, fs_update=1.000, fs_state_norm=14.669, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1152, valid=0.1152, solved=0.1152, blank_acc=0.6345, early=0.6340, late=0.6332, early_late_gap=+0.0008
  future_seed: fs_gate=0.458, fs_update=1.000, fs_state_norm=14.669, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.1855, valid=0.1855, solved=0.1855, blank_acc=0.6471, early=0.6447, late=0.6468, early_late_gap=-0.0021
  future_seed: fs_gate=0.458, fs_update=1.000, fs_state_norm=14.669, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.2031, valid=0.2031, solved=0.2031, blank_acc=0.6479, early=0.6441, late=0.6484, early_late_gap=-0.0043
  future_seed: fs_gate=0.458, fs_update=1.000, fs_state_norm=14.669, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step7000-20260703T0443Z-b040136/runs/gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step7000-20260703T0443Z-b040136/runs/gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 4, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step7000-20260703T0443Z-b040136/runs/gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step7000-20260703T0443Z-b040136/runs/gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.3164, blank_acc=0.7389, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step7000-20260703T0443Z-b040136/runs/gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step7000-20260703T0443Z-b040136/runs/gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0781, blank_acc=0.6033, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.3105, valid=0.3105, solved=0.3105, blank_acc=0.7322, early=0.7406, late=0.7238, early_late_gap=+0.0168
- b56_64 (56-64, n=512): exact=0.0859, valid=0.0859, solved=0.0859, blank_acc=0.5938, early=0.5950, late=0.5935, early_late_gap=+0.0015

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step7000-20260703T0443Z-b040136/runs/gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step7000-20260703T0443Z-b040136/runs/gdn-transition-mixedhard-d224l12-step7000-fix9-20260703T0450Z-b040136/output/case_bank
