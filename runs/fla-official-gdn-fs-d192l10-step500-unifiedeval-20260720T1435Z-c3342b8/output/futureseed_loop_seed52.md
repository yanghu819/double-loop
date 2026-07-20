# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9981, total=1.0058, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0339, loop_last_loss=0.9981, sec=2.0
- loop 1: exact=0.0117, valid=0.0117, solved=0.0117, blank_acc=0.4858, early=0.4823, late=0.4908, early_late_gap=-0.0085
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=22.101, fs_decay=0.00, fs_raw_rms=0.138, fs_raw_rms_std=0.008, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5028, early=0.4978, late=0.5080, early_late_gap=-0.0102
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=22.101, fs_decay=0.00, fs_raw_rms=0.137, fs_raw_rms_std=0.008, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5053, early=0.5006, late=0.5115, early_late_gap=-0.0109
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=22.101, fs_decay=0.00, fs_raw_rms=0.137, fs_raw_rms_std=0.009, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5049, early=0.4998, late=0.5105, early_late_gap=-0.0108
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=22.101, fs_decay=0.00, fs_raw_rms=0.137, fs_raw_rms_std=0.009, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0215, valid=0.0215, solved=0.0215, blank_acc=0.5049, early=0.5002, late=0.5103, early_late_gap=-0.0101
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=22.101, fs_decay=0.00, fs_raw_rms=0.137, fs_raw_rms_std=0.009, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/fla-final-eval-c3342b8/runs/fla-official-gdn-fs-d192l10-step500-unifiedeval-20260720T1435Z-c3342b8/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/fla-final-eval-c3342b8/runs/fla-official-gdn-fs-d192l10-step500-unifiedeval-20260720T1435Z-c3342b8/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.8711, blank_acc=0.9952, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 2}
- official_b51_55: index=/huyang2/double-loop/.worktrees/fla-final-eval-c3342b8/runs/fla-official-gdn-fs-d192l10-step500-unifiedeval-20260720T1435Z-c3342b8/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/fla-final-eval-c3342b8/runs/fla-official-gdn-fs-d192l10-step500-unifiedeval-20260720T1435Z-c3342b8/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5311, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/.worktrees/fla-final-eval-c3342b8/runs/fla-official-gdn-fs-d192l10-step500-unifiedeval-20260720T1435Z-c3342b8/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/fla-final-eval-c3342b8/runs/fla-official-gdn-fs-d192l10-step500-unifiedeval-20260720T1435Z-c3342b8/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4691, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.8594, valid=0.8594, solved=0.8594, blank_acc=0.9947, early=0.9924, late=0.9955, early_late_gap=-0.0031
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5303, early=0.5282, late=0.5268, early_late_gap=+0.0014
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4733, early=0.4673, late=0.4733, early_late_gap=-0.0060

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/fla-final-eval-c3342b8/runs/fla-official-gdn-fs-d192l10-step500-unifiedeval-20260720T1435Z-c3342b8/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/fla-final-eval-c3342b8/runs/fla-official-gdn-fs-d192l10-step500-unifiedeval-20260720T1435Z-c3342b8/output/case_bank
