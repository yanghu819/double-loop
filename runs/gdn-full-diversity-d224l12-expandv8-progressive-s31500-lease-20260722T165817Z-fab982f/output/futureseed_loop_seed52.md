# 9x9 GDN FutureSeed Loop Study

Mainline mechanism: GDN recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.4730, total=0.6059, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.8584, loop_last_loss=0.4730, sec=261.6
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5936, early=0.5948, late=0.5922, early_late_gap=+0.0026
  future_seed: fs_gate=0.406, fs_update=1.000, fs_state_norm=18.371, fs_decay=0.00, fs_raw_rms=0.373, fs_raw_rms_std=0.039, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.1211, valid=0.1211, solved=0.1211, blank_acc=0.7061, early=0.7065, late=0.7034, early_late_gap=+0.0032
  future_seed: fs_gate=0.406, fs_update=1.000, fs_state_norm=18.371, fs_decay=0.00, fs_raw_rms=0.327, fs_raw_rms_std=0.038, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.3730, valid=0.3730, solved=0.3730, blank_acc=0.7624, early=0.7640, late=0.7622, early_late_gap=+0.0017
  future_seed: fs_gate=0.406, fs_update=1.000, fs_state_norm=18.369, fs_decay=0.00, fs_raw_rms=0.347, fs_raw_rms_std=0.063, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.4941, valid=0.4941, solved=0.4941, blank_acc=0.7792, early=0.7808, late=0.7762, early_late_gap=+0.0046
  future_seed: fs_gate=0.406, fs_update=1.000, fs_state_norm=18.360, fs_decay=0.00, fs_raw_rms=0.407, fs_raw_rms_std=0.096, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.5156, valid=0.5156, solved=0.5156, blank_acc=0.7836, early=0.7852, late=0.7811, early_late_gap=+0.0041
  future_seed: fs_gate=0.406, fs_update=1.000, fs_state_norm=18.349, fs_decay=0.00, fs_raw_rms=0.430, fs_raw_rms_std=0.113, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale035-s31500-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s31500-lease-20260722T165817Z-fab982f/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale035-s31500-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s31500-lease-20260722T165817Z-fab982f/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 1, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale035-s31500-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s31500-lease-20260722T165817Z-fab982f/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale035-s31500-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s31500-lease-20260722T165817Z-fab982f/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.6602, blank_acc=0.8602, selected={'solved_by_loop': 4, 'almost_solved': 3, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale035-s31500-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s31500-lease-20260722T165817Z-fab982f/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale035-s31500-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s31500-lease-20260722T165817Z-fab982f/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.3750, blank_acc=0.7126, selected={'solved_by_loop': 4, 'almost_solved': 2, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=512): exact=0.6172, valid=0.6172, solved=0.6172, blank_acc=0.8385, early=0.8387, late=0.8391, early_late_gap=-0.0004
- b56_64 (56-64, n=512): exact=0.3906, valid=0.3906, solved=0.3906, blank_acc=0.7128, early=0.7077, late=0.7091, early_late_gap=-0.0015

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale035-s31500-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s31500-lease-20260722T165817Z-fab982f/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale035-s31500-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s31500-lease-20260722T165817Z-fab982f/output/case_bank
