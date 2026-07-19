# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.5454, total=0.6524, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.8955, loop_last_loss=0.5454, sec=1953.4
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5807, early=0.5821, late=0.5833, early_late_gap=-0.0012
  future_seed: fs_gate=0.432, fs_update=1.000, fs_state_norm=13.831, fs_decay=0.00, fs_raw_rms=0.301, fs_raw_rms_std=0.029, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0586, valid=0.0586, solved=0.0586, blank_acc=0.6794, early=0.6751, late=0.6832, early_late_gap=-0.0082
  future_seed: fs_gate=0.432, fs_update=1.000, fs_state_norm=13.831, fs_decay=0.00, fs_raw_rms=0.263, fs_raw_rms_std=0.030, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.2812, valid=0.2812, solved=0.2812, blank_acc=0.7279, early=0.7273, late=0.7293, early_late_gap=-0.0021
  future_seed: fs_gate=0.432, fs_update=1.000, fs_state_norm=13.831, fs_decay=0.00, fs_raw_rms=0.286, fs_raw_rms_std=0.057, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.3828, valid=0.3828, solved=0.3828, blank_acc=0.7411, early=0.7384, late=0.7419, early_late_gap=-0.0035
  future_seed: fs_gate=0.432, fs_update=1.000, fs_state_norm=13.831, fs_decay=0.00, fs_raw_rms=0.335, fs_raw_rms_std=0.098, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.3945, valid=0.3945, solved=0.3945, blank_acc=0.7432, early=0.7397, late=0.7441, early_late_gap=-0.0044
  future_seed: fs_gate=0.432, fs_update=1.000, fs_state_norm=13.831, fs_decay=0.00, fs_raw_rms=0.352, fs_raw_rms_std=0.113, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale034-upperbound-f8e009b-20260717/runs/gdn-full-diversity-d224l12-upperbound-s16000-lease-20260719T025124Z-f8e009b/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-upperbound-f8e009b-20260717/runs/gdn-full-diversity-d224l12-upperbound-s16000-lease-20260719T025124Z-f8e009b/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 1, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale034-upperbound-f8e009b-20260717/runs/gdn-full-diversity-d224l12-upperbound-s16000-lease-20260719T025124Z-f8e009b/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-upperbound-f8e009b-20260717/runs/gdn-full-diversity-d224l12-upperbound-s16000-lease-20260719T025124Z-f8e009b/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.5586, blank_acc=0.8220, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale034-upperbound-f8e009b-20260717/runs/gdn-full-diversity-d224l12-upperbound-s16000-lease-20260719T025124Z-f8e009b/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale034-upperbound-f8e009b-20260717/runs/gdn-full-diversity-d224l12-upperbound-s16000-lease-20260719T025124Z-f8e009b/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.2656, blank_acc=0.6821, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.5605, valid=0.5625, solved=0.5605, blank_acc=0.8220, early=0.8210, late=0.8226, early_late_gap=-0.0016
- b56_64 (56-64, n=512): exact=0.2598, valid=0.2598, solved=0.2598, blank_acc=0.6782, early=0.6719, late=0.6784, early_late_gap=-0.0065

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale034-upperbound-f8e009b-20260717/runs/gdn-full-diversity-d224l12-upperbound-s16000-lease-20260719T025124Z-f8e009b/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale034-upperbound-f8e009b-20260717/runs/gdn-full-diversity-d224l12-upperbound-s16000-lease-20260719T025124Z-f8e009b/output/case_bank
