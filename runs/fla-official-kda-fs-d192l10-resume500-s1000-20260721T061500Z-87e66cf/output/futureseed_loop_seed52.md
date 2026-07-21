# 9x9 FLA KDA FutureSeed Loop Study

Mainline mechanism: official FLA KDA recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9491, total=0.9564, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9836, loop_last_loss=0.9491, sec=2824.3
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5096, early=0.5059, late=0.5134, early_late_gap=-0.0075
  future_seed: fs_gate=0.506, fs_update=1.000, fs_state_norm=22.882, fs_decay=0.00, fs_raw_rms=0.273, fs_raw_rms_std=0.013, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5227, early=0.5145, late=0.5261, early_late_gap=-0.0115
  future_seed: fs_gate=0.506, fs_update=1.000, fs_state_norm=22.882, fs_decay=0.00, fs_raw_rms=0.265, fs_raw_rms_std=0.015, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5234, early=0.5151, late=0.5275, early_late_gap=-0.0125
  future_seed: fs_gate=0.506, fs_update=1.000, fs_state_norm=22.882, fs_decay=0.00, fs_raw_rms=0.262, fs_raw_rms_std=0.015, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5239, early=0.5167, late=0.5278, early_late_gap=-0.0110
  future_seed: fs_gate=0.506, fs_update=1.000, fs_state_norm=22.882, fs_decay=0.00, fs_raw_rms=0.262, fs_raw_rms_std=0.015, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5238, early=0.5161, late=0.5272, early_late_gap=-0.0111
  future_seed: fs_gate=0.506, fs_update=1.000, fs_state_norm=22.882, fs_decay=0.00, fs_raw_rms=0.261, fs_raw_rms_std=0.016, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/runs/fla-official-kda-fs-d192l10-resume500-s1000-20260721T061500Z-87e66cf/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/runs/fla-official-kda-fs-d192l10-resume500-s1000-20260721T061500Z-87e66cf/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.9961, blank_acc=0.9999, selected={'solved_by_loop': 4, 'almost_solved': 1, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/runs/fla-official-kda-fs-d192l10-resume500-s1000-20260721T061500Z-87e66cf/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/fla-official-kda-fs-d192l10-resume500-s1000-20260721T061500Z-87e66cf/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5449, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/runs/fla-official-kda-fs-d192l10-resume500-s1000-20260721T061500Z-87e66cf/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/runs/fla-official-kda-fs-d192l10-resume500-s1000-20260721T061500Z-87e66cf/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4903, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9998, early=0.9993, late=1.0000, early_late_gap=-0.0007
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5504, early=0.5516, late=0.5479, early_late_gap=+0.0037
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4889, early=0.4831, late=0.4877, early_late_gap=-0.0046

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/fla-official-kda-fs-d192l10-resume500-s1000-20260721T061500Z-87e66cf/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/fla-official-kda-fs-d192l10-resume500-s1000-20260721T061500Z-87e66cf/output/case_bank
