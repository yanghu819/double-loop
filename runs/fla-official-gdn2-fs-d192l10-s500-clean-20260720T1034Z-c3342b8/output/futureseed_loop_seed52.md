# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0101, total=1.0168, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0402, loop_last_loss=1.0101, sec=2895.4
- loop 1: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.4891, early=0.4857, late=0.4927, early_late_gap=-0.0070
  future_seed: fs_gate=0.503, fs_update=1.000, fs_state_norm=22.778, fs_decay=0.00, fs_raw_rms=0.491, fs_raw_rms_std=0.033, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5004, early=0.4955, late=0.5052, early_late_gap=-0.0097
  future_seed: fs_gate=0.503, fs_update=1.000, fs_state_norm=22.778, fs_decay=0.00, fs_raw_rms=0.470, fs_raw_rms_std=0.035, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5008, early=0.4976, late=0.5038, early_late_gap=-0.0062
  future_seed: fs_gate=0.503, fs_update=1.000, fs_state_norm=22.778, fs_decay=0.00, fs_raw_rms=0.462, fs_raw_rms_std=0.035, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5015, early=0.4977, late=0.5042, early_late_gap=-0.0066
  future_seed: fs_gate=0.503, fs_update=1.000, fs_state_norm=22.778, fs_decay=0.00, fs_raw_rms=0.460, fs_raw_rms_std=0.035, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.5012, early=0.4973, late=0.5040, early_late_gap=-0.0068
  future_seed: fs_gate=0.503, fs_update=1.000, fs_state_norm=22.778, fs_decay=0.00, fs_raw_rms=0.459, fs_raw_rms_std=0.035, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-gdn2-fs-d192l10-s500-clean-20260720T1034Z-c3342b8/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-gdn2-fs-d192l10-s500-clean-20260720T1034Z-c3342b8/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.7266, blank_acc=0.9889, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 2}
- official_b51_55: index=/huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-gdn2-fs-d192l10-s500-clean-20260720T1034Z-c3342b8/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-gdn2-fs-d192l10-s500-clean-20260720T1034Z-c3342b8/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5209, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-gdn2-fs-d192l10-s500-clean-20260720T1034Z-c3342b8/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-gdn2-fs-d192l10-s500-clean-20260720T1034Z-c3342b8/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4696, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.7539, valid=0.7539, solved=0.7539, blank_acc=0.9903, early=0.9856, late=0.9929, early_late_gap=-0.0073
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5218, early=0.5129, late=0.5207, early_late_gap=-0.0078
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4685, early=0.4613, late=0.4707, early_late_gap=-0.0093

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-gdn2-fs-d192l10-s500-clean-20260720T1034Z-c3342b8/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-gdn2-fs-d192l10-s500-clean-20260720T1034Z-c3342b8/output/case_bank
