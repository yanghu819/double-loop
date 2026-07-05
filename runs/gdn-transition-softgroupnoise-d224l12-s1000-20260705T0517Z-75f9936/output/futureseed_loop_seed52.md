# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9906, total=1.0020, loop_loss=all, noise=hidden_aggregate, buffer=8192, dtype=float32, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0414, loop_last_loss=0.9906, sec=2913.9
- loop 1: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5003, early=0.4979, late=0.5042, early_late_gap=-0.0063
  future_seed: fs_gate=0.487, fs_update=1.000, fs_state_norm=15.590, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5206, early=0.5242, late=0.5176, early_late_gap=+0.0066
  future_seed: fs_gate=0.487, fs_update=1.000, fs_state_norm=15.590, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5239, early=0.5272, late=0.5222, early_late_gap=+0.0050
  future_seed: fs_gate=0.487, fs_update=1.000, fs_state_norm=15.590, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5252, early=0.5283, late=0.5233, early_late_gap=+0.0050
  future_seed: fs_gate=0.487, fs_update=1.000, fs_state_norm=15.590, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5252, early=0.5292, late=0.5228, early_late_gap=+0.0064
  future_seed: fs_gate=0.487, fs_update=1.000, fs_state_norm=15.590, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0234, oracle_solved=0.0234, disagree=0.0000
  confidence: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5252, early=0.5292, late=0.5228, early_late_gap=+0.0064; gap=0.0000
  consistency: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5252, early=0.5292, late=0.5228, early_late_gap=+0.0064; gap=0.0000
  residual: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5252, early=0.5292, late=0.5228, early_late_gap=+0.0064; gap=0.0000
  majority: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5252, early=0.5292, late=0.5228, early_late_gap=+0.0064; gap=0.0000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pdiag024-softgroup-75f9936/runs/gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-75f9936/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pdiag024-softgroup-75f9936/runs/gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-75f9936/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.9961, blank_acc=0.9998, selected={'solved_by_loop': 4, 'almost_solved': 1, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pdiag024-softgroup-75f9936/runs/gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-75f9936/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pdiag024-softgroup-75f9936/runs/gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-75f9936/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5338, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pdiag024-softgroup-75f9936/runs/gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-75f9936/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pdiag024-softgroup-75f9936/runs/gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-75f9936/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4897, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5478, early=0.5473, late=0.5477, early_late_gap=-0.0004
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4871, early=0.4892, late=0.4807, early_late_gap=+0.0085

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pdiag024-softgroup-75f9936/runs/gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-75f9936/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pdiag024-softgroup-75f9936/runs/gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-75f9936/output/case_bank
