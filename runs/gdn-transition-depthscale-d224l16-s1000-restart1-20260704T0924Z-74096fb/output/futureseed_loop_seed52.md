# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9999, total=1.0100, loop_loss=all, noise=feature_diff, buffer=8192, dtype=float32, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0459, loop_last_loss=0.9999, sec=3922.9
- loop 1: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.4986, early=0.5003, late=0.4952, early_late_gap=+0.0051
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=15.608, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5172, early=0.5210, late=0.5075, early_late_gap=+0.0135
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=15.607, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5195, early=0.5235, late=0.5108, early_late_gap=+0.0128
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=15.607, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5195, early=0.5238, late=0.5108, early_late_gap=+0.0131
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=15.607, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5192, early=0.5235, late=0.5109, early_late_gap=+0.0127
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=15.607, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb/runs/gdn-transition-depthscale-d224l16-s1000-restart1-20260704T0924Z-74096fb/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb/runs/gdn-transition-depthscale-d224l16-s1000-restart1-20260704T0924Z-74096fb/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.9766, blank_acc=0.9990, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb/runs/gdn-transition-depthscale-d224l16-s1000-restart1-20260704T0924Z-74096fb/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb/runs/gdn-transition-depthscale-d224l16-s1000-restart1-20260704T0924Z-74096fb/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5354, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb/runs/gdn-transition-depthscale-d224l16-s1000-restart1-20260704T0924Z-74096fb/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb/runs/gdn-transition-depthscale-d224l16-s1000-restart1-20260704T0924Z-74096fb/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4870, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.9922, valid=0.9922, solved=0.9922, blank_acc=0.9995, early=0.9993, late=0.9996, early_late_gap=-0.0004
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5445, early=0.5455, late=0.5389, early_late_gap=+0.0066
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4836, early=0.4848, late=0.4705, early_late_gap=+0.0143

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb/runs/gdn-transition-depthscale-d224l16-s1000-restart1-20260704T0924Z-74096fb/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-transition-depthscale-d224l16-s4000-20260704T0841Z-74096fb/runs/gdn-transition-depthscale-d224l16-s1000-restart1-20260704T0924Z-74096fb/output/case_bank
