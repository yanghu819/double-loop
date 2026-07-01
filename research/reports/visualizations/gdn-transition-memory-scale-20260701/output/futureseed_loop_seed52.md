# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.8863, total=0.8863, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.1910, loop_last_loss=0.8863, sec=3973.8
- loop 1: exact=0.0020, valid=0.0020, solved=0.0020, blank_acc=0.4425, early=0.4410, late=0.4344, early_late_gap=+0.0066
  future_seed: fs_gate=0.478, fs_update=1.000, fs_state_norm=15.296, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0273, valid=0.0273, solved=0.0273, blank_acc=0.5054, early=0.5034, late=0.5086, early_late_gap=-0.0052
  future_seed: fs_gate=0.478, fs_update=1.000, fs_state_norm=15.296, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0283, valid=0.0283, solved=0.0283, blank_acc=0.5447, early=0.5455, late=0.5472, early_late_gap=-0.0017
  future_seed: fs_gate=0.478, fs_update=1.000, fs_state_norm=15.296, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0381, valid=0.0381, solved=0.0381, blank_acc=0.5564, early=0.5529, late=0.5572, early_late_gap=-0.0043
  future_seed: fs_gate=0.478, fs_update=1.000, fs_state_norm=15.296, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0381, valid=0.0381, solved=0.0381, blank_acc=0.5571, early=0.5542, late=0.5576, early_late_gap=-0.0034
  future_seed: fs_gate=0.478, fs_update=1.000, fs_state_norm=15.296, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/gdn-transition-memory-29c8aea-20260701T0301Z/runs/gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-memory-29c8aea-20260701T0301Z/runs/gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=1.0000, blank_acc=1.0000, selected={'solved_by_loop': 2, 'almost_solved': 0, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/gdn-transition-memory-29c8aea-20260701T0301Z/runs/gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-memory-29c8aea-20260701T0301Z/runs/gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0039, blank_acc=0.5916, selected={'solved_by_loop': 1, 'almost_solved': 1, 'hard_failure': 2}
- official_b56_64: index=/huyang2/double-loop/.worktrees/gdn-transition-memory-29c8aea-20260701T0301Z/runs/gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-memory-29c8aea-20260701T0301Z/runs/gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0039, blank_acc=0.5039, selected={'solved_by_loop': 1, 'almost_solved': 1, 'hard_failure': 2}

## Official Blank-Range Eval

- b46_50 (46-50, n=1024): exact=1.0000, valid=1.0000, solved=1.0000, blank_acc=1.0000, early=1.0000, late=1.0000, early_late_gap=+0.0000
- b51_55 (51-55, n=1024): exact=0.0098, valid=0.0098, solved=0.0098, blank_acc=0.6019, early=0.5981, late=0.6020, early_late_gap=-0.0039
- b56_64 (56-64, n=1024): exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.5157, early=0.5128, late=0.5156, early_late_gap=-0.0028

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-transition-memory-29c8aea-20260701T0301Z/runs/gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-transition-memory-29c8aea-20260701T0301Z/runs/gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea/output/case_bank
