# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.3467, total=0.4890, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.8379, loop_last_loss=0.3467, sec=8070.6
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5370, early=0.5303, late=0.5402, early_late_gap=-0.0099
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.797, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0352, valid=0.0352, solved=0.0352, blank_acc=0.5881, early=0.5890, late=0.5864, early_late_gap=+0.0026
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.797, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1289, valid=0.1289, solved=0.1289, blank_acc=0.6136, early=0.6112, late=0.6142, early_late_gap=-0.0030
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.797, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.2031, valid=0.2031, solved=0.2031, blank_acc=0.6214, early=0.6187, late=0.6208, early_late_gap=-0.0021
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.797, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.2070, valid=0.2070, solved=0.2070, blank_acc=0.6216, early=0.6184, late=0.6196, early_late_gap=-0.0012
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.797, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 3, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.3333, blank_acc=0.7392, selected={'solved_by_loop': 3, 'almost_solved': 3, 'hard_failure': 3}
- official_b56_64: index=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0990, blank_acc=0.5612, selected={'solved_by_loop': 3, 'almost_solved': 2, 'hard_failure': 3}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.2832, valid=0.2832, solved=0.2832, blank_acc=0.7139, early=0.7204, late=0.7135, early_late_gap=+0.0069
- b56_64 (56-64, n=512): exact=0.0801, valid=0.0801, solved=0.0801, blank_acc=0.5405, early=0.5371, late=0.5378, early_late_gap=-0.0007

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/output/case_bank
