# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0133, total=1.0133, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.1348, loop_last_loss=1.0133, sec=1409.1
- loop 1: exact=0.0083, valid=0.0083, solved=0.0083, blank_acc=0.4651, early=0.4651, late=0.4697, early_late_gap=-0.0046
  future_seed: fs_gate=0.496, fs_update=1.000, fs_state_norm=7.943, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0288, valid=0.0288, solved=0.0288, blank_acc=0.5048, early=0.4986, late=0.5131, early_late_gap=-0.0145
  future_seed: fs_gate=0.496, fs_update=1.000, fs_state_norm=7.943, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5215, early=0.5173, late=0.5276, early_late_gap=-0.0103
  future_seed: fs_gate=0.496, fs_update=1.000, fs_state_norm=7.943, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5223, early=0.5182, late=0.5281, early_late_gap=-0.0099
  future_seed: fs_gate=0.496, fs_update=1.000, fs_state_norm=7.943, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5225, early=0.5181, late=0.5285, early_late_gap=-0.0104
  future_seed: fs_gate=0.496, fs_update=1.000, fs_state_norm=7.943, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5226, early=0.5180, late=0.5288, early_late_gap=-0.0109
  future_seed: fs_gate=0.496, fs_update=1.000, fs_state_norm=7.943, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 7: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5226, early=0.5179, late=0.5290, early_late_gap=-0.0111
  future_seed: fs_gate=0.496, fs_update=1.000, fs_state_norm=7.943, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 8: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5226, early=0.5180, late=0.5288, early_late_gap=-0.0108
  future_seed: fs_gate=0.496, fs_update=1.000, fs_state_norm=7.943, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official: index=/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-loop8-dynamics-s1500-83b6bff-20260629/runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff/output/case_bank/official/index.html; cases=/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-loop8-dynamics-s1500-83b6bff-20260629/runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff/output/case_bank/official/cases.json; final_loop=8, exact=0.0352, blank_acc=0.5223, selected={'solved_by_loop': 3, 'almost_solved': 0, 'hard_failure': 3}

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-loop8-dynamics-s1500-83b6bff-20260629/runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-loop8-dynamics-s1500-83b6bff-20260629/runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff/output/case_bank
