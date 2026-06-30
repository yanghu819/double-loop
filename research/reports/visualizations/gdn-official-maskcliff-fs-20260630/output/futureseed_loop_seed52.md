# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0091, total=1.0091, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.1829, loop_last_loss=1.0091, sec=616.6
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4667, early=0.4595, late=0.4683, early_late_gap=-0.0087
  future_seed: fs_gate=0.490, fs_update=1.000, fs_state_norm=7.845, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5019, early=0.4986, late=0.5083, early_late_gap=-0.0097
  future_seed: fs_gate=0.490, fs_update=1.000, fs_state_norm=7.845, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5233, early=0.5160, late=0.5273, early_late_gap=-0.0113
  future_seed: fs_gate=0.490, fs_update=1.000, fs_state_norm=7.845, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5252, early=0.5193, late=0.5274, early_late_gap=-0.0081
  future_seed: fs_gate=0.490, fs_update=1.000, fs_state_norm=7.845, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5252, early=0.5190, late=0.5283, early_late_gap=-0.0092
  future_seed: fs_gate=0.490, fs_update=1.000, fs_state_norm=7.845, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/gdn-maskcliff-74ed759-20260630T1357Z/runs/gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/gdn-maskcliff-74ed759-20260630T1357Z/runs/gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.9805, blank_acc=0.9985, selected={'solved_by_loop': 1, 'almost_solved': 1, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/gdn-maskcliff-74ed759-20260630T1357Z/runs/gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/gdn-maskcliff-74ed759-20260630T1357Z/runs/gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5336, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 1}
- official_b56_64: index=/huyang2/double-loop/.worktrees/gdn-maskcliff-74ed759-20260630T1357Z/runs/gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/gdn-maskcliff-74ed759-20260630T1357Z/runs/gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4886, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 1}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.9863, valid=0.9863, solved=0.9863, blank_acc=0.9993, early=0.9989, late=0.9996, early_late_gap=-0.0007
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5457, early=0.5412, late=0.5389, early_late_gap=+0.0023
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4879, early=0.4883, late=0.4798, early_late_gap=+0.0086

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-maskcliff-74ed759-20260630T1357Z/runs/gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-maskcliff-74ed759-20260630T1357Z/runs/gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759/output/case_bank
