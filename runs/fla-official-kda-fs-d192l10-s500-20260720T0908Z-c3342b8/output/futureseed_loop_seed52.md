# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0252, total=1.0301, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0483, loop_last_loss=1.0252, sec=3236.9
- loop 1: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.4854, early=0.4801, late=0.4836, early_late_gap=-0.0035
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=22.564, fs_decay=0.00, fs_raw_rms=0.239, fs_raw_rms_std=0.011, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0117, valid=0.0117, solved=0.0117, blank_acc=0.4962, early=0.4905, late=0.4967, early_late_gap=-0.0062
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=22.564, fs_decay=0.00, fs_raw_rms=0.234, fs_raw_rms_std=0.012, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0137, valid=0.0137, solved=0.0137, blank_acc=0.4965, early=0.4912, late=0.5004, early_late_gap=-0.0092
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=22.564, fs_decay=0.00, fs_raw_rms=0.231, fs_raw_rms_std=0.012, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.4969, early=0.4911, late=0.5010, early_late_gap=-0.0099
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=22.564, fs_decay=0.00, fs_raw_rms=0.231, fs_raw_rms_std=0.012, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.4971, early=0.4910, late=0.5018, early_late_gap=-0.0108
  future_seed: fs_gate=0.499, fs_update=1.000, fs_state_norm=22.564, fs_decay=0.00, fs_raw_rms=0.231, fs_raw_rms_std=0.012, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-kda-fs-d192l10-s500-20260720T0908Z-c3342b8/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-kda-fs-d192l10-s500-20260720T0908Z-c3342b8/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.6523, blank_acc=0.9866, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 3}
- official_b51_55: index=/huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-kda-fs-d192l10-s500-20260720T0908Z-c3342b8/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-kda-fs-d192l10-s500-20260720T0908Z-c3342b8/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5176, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-kda-fs-d192l10-s500-20260720T0908Z-c3342b8/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-kda-fs-d192l10-s500-20260720T0908Z-c3342b8/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4595, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.6504, valid=0.6504, solved=0.6504, blank_acc=0.9855, early=0.9844, late=0.9865, early_late_gap=-0.0021
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5220, early=0.5235, late=0.5202, early_late_gap=+0.0033
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4587, early=0.4556, late=0.4592, early_late_gap=-0.0036

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-kda-fs-d192l10-s500-20260720T0908Z-c3342b8/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/fla-official-7891539-20260720/runs/fla-official-kda-fs-d192l10-s500-20260720T0908Z-c3342b8/output/case_bank
