# 9x9 FLA GDN2 FutureSeed Loop Study

Mainline mechanism: FLA GDN2 recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9539, total=0.9606, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9837, loop_last_loss=0.9539, sec=2280.8
- loop 1: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5096, early=0.5048, late=0.5121, early_late_gap=-0.0073
  future_seed: fs_gate=0.508, fs_update=1.000, fs_state_norm=22.977, fs_decay=0.00, fs_raw_rms=0.656, fs_raw_rms_std=0.054, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5262, early=0.5234, late=0.5312, early_late_gap=-0.0078
  future_seed: fs_gate=0.508, fs_update=1.000, fs_state_norm=22.977, fs_decay=0.00, fs_raw_rms=0.574, fs_raw_rms_std=0.055, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5266, early=0.5221, late=0.5325, early_late_gap=-0.0104
  future_seed: fs_gate=0.508, fs_update=1.000, fs_state_norm=22.977, fs_decay=0.00, fs_raw_rms=0.546, fs_raw_rms_std=0.054, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5269, early=0.5215, late=0.5343, early_late_gap=-0.0127
  future_seed: fs_gate=0.508, fs_update=1.000, fs_state_norm=22.977, fs_decay=0.00, fs_raw_rms=0.539, fs_raw_rms_std=0.054, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5280, early=0.5223, late=0.5349, early_late_gap=-0.0126
  future_seed: fs_gate=0.508, fs_update=1.000, fs_state_norm=22.977, fs_decay=0.00, fs_raw_rms=0.537, fs_raw_rms_std=0.054, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/runs/fla-official-gdn2-fs-d192l10-resume600-s1000-direct-leg2-20260721T045147Z-87e66cf/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/runs/fla-official-gdn2-fs-d192l10-resume600-s1000-direct-leg2-20260721T045147Z-87e66cf/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.9883, blank_acc=0.9995, selected={'solved_by_loop': 4, 'almost_solved': 3, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/runs/fla-official-gdn2-fs-d192l10-resume600-s1000-direct-leg2-20260721T045147Z-87e66cf/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/fla-official-gdn2-fs-d192l10-resume600-s1000-direct-leg2-20260721T045147Z-87e66cf/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5409, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/runs/fla-official-gdn2-fs-d192l10-resume600-s1000-direct-leg2-20260721T045147Z-87e66cf/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/runs/fla-official-gdn2-fs-d192l10-resume600-s1000-direct-leg2-20260721T045147Z-87e66cf/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4907, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.9922, valid=0.9922, solved=0.9922, blank_acc=0.9997, early=0.9993, late=0.9999, early_late_gap=-0.0006
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5506, early=0.5487, late=0.5489, early_late_gap=-0.0001
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4858, early=0.4864, late=0.4816, early_late_gap=+0.0048

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/fla-official-gdn2-fs-d192l10-resume600-s1000-direct-leg2-20260721T045147Z-87e66cf/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/fla-official-gdn2-fs-d192l10-resume600-s1000-direct-leg2-20260721T045147Z-87e66cf/output/case_bank
