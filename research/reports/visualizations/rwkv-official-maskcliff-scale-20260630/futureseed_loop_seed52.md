# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9882, total=0.9882, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.3881, loop_last_loss=0.9882, sec=2791.0
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4408, early=0.4228, late=0.4511, early_late_gap=-0.0282
  future_seed: fs_gate=0.537, fs_update=1.000, fs_state_norm=8.585, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.4976, early=0.4873, late=0.5071, early_late_gap=-0.0197
  future_seed: fs_gate=0.537, fs_update=1.000, fs_state_norm=8.585, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5285, early=0.5255, late=0.5332, early_late_gap=-0.0077
  future_seed: fs_gate=0.537, fs_update=1.000, fs_state_norm=8.585, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5305, early=0.5278, late=0.5359, early_late_gap=-0.0081
  future_seed: fs_gate=0.537, fs_update=1.000, fs_state_norm=8.585, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5306, early=0.5272, late=0.5353, early_late_gap=-0.0081
  future_seed: fs_gate=0.537, fs_update=1.000, fs_state_norm=8.585, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5310, early=0.5280, late=0.5355, early_late_gap=-0.0075
  future_seed: fs_gate=0.537, fs_update=1.000, fs_state_norm=8.585, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 7: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5310, early=0.5283, late=0.5353, early_late_gap=-0.0070
  future_seed: fs_gate=0.537, fs_update=1.000, fs_state_norm=8.585, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 8: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5310, early=0.5286, late=0.5351, early_late_gap=-0.0065
  future_seed: fs_gate=0.537, fs_update=1.000, fs_state_norm=8.585, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/rwkv-maskcliff-83ccd29-20260630T1231Z/runs/rwkv-official-maskcliff-d256l12-s2000-20260630T1240Z-83ccd29/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/rwkv-maskcliff-83ccd29-20260630T1231Z/runs/rwkv-official-maskcliff-d256l12-s2000-20260630T1240Z-83ccd29/output/case_bank/official_b46_50/cases.json; final_loop=8, exact=0.9922, blank_acc=0.9998, selected={'solved_by_loop': 2, 'almost_solved': 2, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/rwkv-maskcliff-83ccd29-20260630T1231Z/runs/rwkv-official-maskcliff-d256l12-s2000-20260630T1240Z-83ccd29/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/rwkv-maskcliff-83ccd29-20260630T1231Z/runs/rwkv-official-maskcliff-d256l12-s2000-20260630T1240Z-83ccd29/output/case_bank/official_b51_55/cases.json; final_loop=8, exact=0.0000, blank_acc=0.5326, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 2}
- official_b56_64: index=/huyang2/double-loop/.worktrees/rwkv-maskcliff-83ccd29-20260630T1231Z/runs/rwkv-official-maskcliff-d256l12-s2000-20260630T1240Z-83ccd29/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/rwkv-maskcliff-83ccd29-20260630T1231Z/runs/rwkv-official-maskcliff-d256l12-s2000-20260630T1240Z-83ccd29/output/case_bank/official_b56_64/cases.json; final_loop=8, exact=0.0000, blank_acc=0.4898, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 2}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.9883, valid=0.9883, solved=0.9883, blank_acc=0.9993, early=0.9993, late=0.9994, early_late_gap=-0.0001
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5432, early=0.5421, late=0.5416, early_late_gap=+0.0005
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4893, early=0.4888, late=0.4861, early_late_gap=+0.0027

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/rwkv-maskcliff-83ccd29-20260630T1231Z/runs/rwkv-official-maskcliff-d256l12-s2000-20260630T1240Z-83ccd29/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/rwkv-maskcliff-83ccd29-20260630T1231Z/runs/rwkv-official-maskcliff-d256l12-s2000-20260630T1240Z-83ccd29/output/case_bank
