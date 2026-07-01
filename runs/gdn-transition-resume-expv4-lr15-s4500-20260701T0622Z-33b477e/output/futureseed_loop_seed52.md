# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.6765, total=0.6765, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.1167, loop_last_loss=0.6765, sec=2079.8
- loop 1: exact=0.0088, valid=0.0088, solved=0.0088, blank_acc=0.4558, early=0.4564, late=0.4494, early_late_gap=+0.0070
  future_seed: fs_gate=0.473, fs_update=1.000, fs_state_norm=15.146, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0283, valid=0.0283, solved=0.0283, blank_acc=0.5101, early=0.5103, late=0.5110, early_late_gap=-0.0007
  future_seed: fs_gate=0.473, fs_update=1.000, fs_state_norm=15.146, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5563, early=0.5538, late=0.5590, early_late_gap=-0.0052
  future_seed: fs_gate=0.473, fs_update=1.000, fs_state_norm=15.146, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0449, valid=0.0449, solved=0.0449, blank_acc=0.5794, early=0.5778, late=0.5825, early_late_gap=-0.0047
  future_seed: fs_gate=0.473, fs_update=1.000, fs_state_norm=15.146, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0537, valid=0.0537, solved=0.0537, blank_acc=0.5823, early=0.5805, late=0.5861, early_late_gap=-0.0056
  future_seed: fs_gate=0.473, fs_update=1.000, fs_state_norm=15.146, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/gdn-transition-resume-33b477e-20260701T0620Z/runs/gdn-transition-resume-expv4-lr15-s4500-20260701T0622Z-33b477e/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-resume-33b477e-20260701T0620Z/runs/gdn-transition-resume-expv4-lr15-s4500-20260701T0622Z-33b477e/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 2, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/gdn-transition-resume-33b477e-20260701T0620Z/runs/gdn-transition-resume-expv4-lr15-s4500-20260701T0622Z-33b477e/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-resume-33b477e-20260701T0620Z/runs/gdn-transition-resume-expv4-lr15-s4500-20260701T0622Z-33b477e/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0195, blank_acc=0.6367, selected={'solved_by_loop': 2, 'almost_solved': 2, 'hard_failure': 2}
- official_b56_64: index=/huyang2/double-loop/.worktrees/gdn-transition-resume-33b477e-20260701T0620Z/runs/gdn-transition-resume-expv4-lr15-s4500-20260701T0622Z-33b477e/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-resume-33b477e-20260701T0620Z/runs/gdn-transition-resume-expv4-lr15-s4500-20260701T0622Z-33b477e/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0234, blank_acc=0.5248, selected={'solved_by_loop': 2, 'almost_solved': 2, 'hard_failure': 2}

## Official Blank-Range Eval

- b46_50 (46-50, n=1024): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=1024): exact=0.0459, valid=0.0459, solved=0.0459, blank_acc=0.6555, early=0.6470, late=0.6577, early_late_gap=-0.0107
- b56_64 (56-64, n=1024): exact=0.0332, valid=0.0332, solved=0.0332, blank_acc=0.5334, early=0.5335, late=0.5352, early_late_gap=-0.0017

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-transition-resume-33b477e-20260701T0620Z/runs/gdn-transition-resume-expv4-lr15-s4500-20260701T0622Z-33b477e/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-transition-resume-33b477e-20260701T0620Z/runs/gdn-transition-resume-expv4-lr15-s4500-20260701T0622Z-33b477e/output/case_bank
