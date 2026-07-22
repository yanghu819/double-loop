# 9x9 GDN FutureSeed Loop Study

Mainline mechanism: GDN recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.4787, total=0.6097, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.8626, loop_last_loss=0.4787, sec=946.2
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5998, early=0.6040, late=0.5951, early_late_gap=+0.0088
  future_seed: fs_gate=0.404, fs_update=1.000, fs_state_norm=18.289, fs_decay=0.00, fs_raw_rms=0.393, fs_raw_rms_std=0.042, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.1074, valid=0.1074, solved=0.1074, blank_acc=0.7019, early=0.6994, late=0.7025, early_late_gap=-0.0031
  future_seed: fs_gate=0.404, fs_update=1.000, fs_state_norm=18.289, fs_decay=0.00, fs_raw_rms=0.337, fs_raw_rms_std=0.037, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.3594, valid=0.3594, solved=0.3594, blank_acc=0.7440, early=0.7461, late=0.7416, early_late_gap=+0.0045
  future_seed: fs_gate=0.404, fs_update=1.000, fs_state_norm=18.289, fs_decay=0.00, fs_raw_rms=0.361, fs_raw_rms_std=0.061, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.4375, valid=0.4375, solved=0.4375, blank_acc=0.7609, early=0.7583, late=0.7596, early_late_gap=-0.0013
  future_seed: fs_gate=0.404, fs_update=1.000, fs_state_norm=18.283, fs_decay=0.00, fs_raw_rms=0.435, fs_raw_rms_std=0.104, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.4668, valid=0.4668, solved=0.4668, blank_acc=0.7664, early=0.7625, late=0.7666, early_late_gap=-0.0041
  future_seed: fs_gate=0.404, fs_update=1.000, fs_state_norm=18.267, fs_decay=0.00, fs_raw_rms=0.457, fs_raw_rms_std=0.130, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/pscale035-s33000-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s33000-lease-20260722T210435Z-fab982f/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/pscale035-s33000-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s33000-lease-20260722T210435Z-fab982f/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.9961, blank_acc=0.9999, selected={'solved_by_loop': 0, 'almost_solved': 1, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/pscale035-s33000-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s33000-lease-20260722T210435Z-fab982f/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/pscale035-s33000-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s33000-lease-20260722T210435Z-fab982f/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.6094, blank_acc=0.8521, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/pscale035-s33000-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s33000-lease-20260722T210435Z-fab982f/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/pscale035-s33000-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s33000-lease-20260722T210435Z-fab982f/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.3438, blank_acc=0.7054, selected={'solved_by_loop': 4, 'almost_solved': 1, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9999, early=0.9999, late=1.0000, early_late_gap=-0.0001
- b51_55 (51-55, n=512): exact=0.6270, valid=0.6270, solved=0.6270, blank_acc=0.8444, early=0.8413, late=0.8469, early_late_gap=-0.0056
- b56_64 (56-64, n=512): exact=0.3633, valid=0.3633, solved=0.3633, blank_acc=0.7045, early=0.7004, late=0.7032, early_late_gap=-0.0028

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/pscale035-s33000-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s33000-lease-20260722T210435Z-fab982f/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/pscale035-s33000-fab982f-full-20260722/runs/gdn-full-diversity-d224l12-expandv8-progressive-s33000-lease-20260722T210435Z-fab982f/output/case_bank
