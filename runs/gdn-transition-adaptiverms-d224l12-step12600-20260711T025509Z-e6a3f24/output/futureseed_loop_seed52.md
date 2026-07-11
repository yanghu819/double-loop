# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0297, total=0.2240, loop_loss=all, noise=none, buffer=8192, dtype=float32, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.8452, loop_last_loss=0.0297, sec=0.6
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5387, early=0.5402, late=0.5354, early_late_gap=+0.0047
  future_seed: fs_gate=0.436, fs_update=1.000, fs_state_norm=13.855, fs_decay=0.00, fs_raw_rms=0.266, fs_raw_rms_std=0.027, fs_norm_gain=0.988, fs_norm_gain_std=0.003, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0566, valid=0.0566, solved=0.0566, blank_acc=0.5918, early=0.5869, late=0.5955, early_late_gap=-0.0086
  future_seed: fs_gate=0.436, fs_update=1.000, fs_state_norm=13.848, fs_decay=0.00, fs_raw_rms=0.285, fs_raw_rms_std=0.026, fs_norm_gain=0.988, fs_norm_gain_std=0.002, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1484, valid=0.1504, solved=0.1484, blank_acc=0.6139, early=0.6092, late=0.6131, early_late_gap=-0.0039
  future_seed: fs_gate=0.436, fs_update=1.000, fs_state_norm=13.839, fs_decay=0.00, fs_raw_rms=0.350, fs_raw_rms_std=0.035, fs_norm_gain=0.988, fs_norm_gain_std=0.002, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.2051, valid=0.2070, solved=0.2051, blank_acc=0.6259, early=0.6232, late=0.6238, early_late_gap=-0.0006
  future_seed: fs_gate=0.436, fs_update=1.000, fs_state_norm=13.839, fs_decay=0.00, fs_raw_rms=0.384, fs_raw_rms_std=0.046, fs_norm_gain=0.988, fs_norm_gain_std=0.003, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.2109, valid=0.2129, solved=0.2109, blank_acc=0.6260, early=0.6219, late=0.6244, early_late_gap=-0.0024
  future_seed: fs_gate=0.436, fs_update=1.000, fs_state_norm=13.840, fs_decay=0.00, fs_raw_rms=0.390, fs_raw_rms_std=0.050, fs_norm_gain=0.988, fs_norm_gain_std=0.003, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pdiag027-adaptiverms-e6a3f24-20260711T025509Z/runs/gdn-transition-adaptiverms-d224l12-step12600-20260711T025509Z-e6a3f24/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pdiag027-adaptiverms-e6a3f24-20260711T025509Z/runs/gdn-transition-adaptiverms-d224l12-step12600-20260711T025509Z-e6a3f24/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 4, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pdiag027-adaptiverms-e6a3f24-20260711T025509Z/runs/gdn-transition-adaptiverms-d224l12-step12600-20260711T025509Z-e6a3f24/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pdiag027-adaptiverms-e6a3f24-20260711T025509Z/runs/gdn-transition-adaptiverms-d224l12-step12600-20260711T025509Z-e6a3f24/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.3086, blank_acc=0.7070, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pdiag027-adaptiverms-e6a3f24-20260711T025509Z/runs/gdn-transition-adaptiverms-d224l12-step12600-20260711T025509Z-e6a3f24/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pdiag027-adaptiverms-e6a3f24-20260711T025509Z/runs/gdn-transition-adaptiverms-d224l12-step12600-20260711T025509Z-e6a3f24/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.1172, blank_acc=0.5639, selected={'solved_by_loop': 4, 'almost_solved': 2, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.3418, valid=0.3418, solved=0.3418, blank_acc=0.7171, early=0.7186, late=0.7188, early_late_gap=-0.0003
- b56_64 (56-64, n=512): exact=0.1367, valid=0.1367, solved=0.1367, blank_acc=0.5481, early=0.5502, late=0.5503, early_late_gap=-0.0001

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pdiag027-adaptiverms-e6a3f24-20260711T025509Z/runs/gdn-transition-adaptiverms-d224l12-step12600-20260711T025509Z-e6a3f24/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pdiag027-adaptiverms-e6a3f24-20260711T025509Z/runs/gdn-transition-adaptiverms-d224l12-step12600-20260711T025509Z-e6a3f24/output/case_bank
