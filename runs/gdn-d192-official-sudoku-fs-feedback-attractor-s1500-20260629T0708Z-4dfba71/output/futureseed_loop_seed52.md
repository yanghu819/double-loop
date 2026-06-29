# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9868, total=0.9868, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=1.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.1761, loop_last_loss=0.9868, sec=989.6
- loop 1: exact=0.0063, valid=0.0063, solved=0.0063, blank_acc=0.4594, early=0.4525, late=0.4607, early_late_gap=-0.0081
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=7.811, fs_decay=0.00, fb_in=0.000, fb_next=1.727, fb_corrupt=0.000, fb_conf=0.708, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5188, early=0.5172, late=0.5217, early_late_gap=-0.0045
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=7.811, fs_decay=0.00, fb_in=1.727, fb_next=1.714, fb_corrupt=0.000, fb_conf=0.676, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5297, early=0.5286, late=0.5333, early_late_gap=-0.0048
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=7.811, fs_decay=0.00, fb_in=1.714, fb_next=1.714, fb_corrupt=0.000, fb_conf=0.673, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5307, early=0.5294, late=0.5348, early_late_gap=-0.0053
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=7.811, fs_decay=0.00, fb_in=1.714, fb_next=1.715, fb_corrupt=0.000, fb_conf=0.672, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5308, early=0.5296, late=0.5348, early_late_gap=-0.0053
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=7.811, fs_decay=0.00, fb_in=1.715, fb_next=1.715, fb_corrupt=0.000, fb_conf=0.672, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official: index=/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-feedback-attractor-s1500-4dfba71-20260629/runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71/output/case_bank/official/index.html; cases=/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-feedback-attractor-s1500-4dfba71-20260629/runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71/output/case_bank/official/cases.json; final_loop=5, exact=0.0352, blank_acc=0.5307, selected={'solved_by_loop': 3, 'almost_solved': 0, 'hard_failure': 3}

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-feedback-attractor-s1500-4dfba71-20260629/runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-feedback-attractor-s1500-4dfba71-20260629/runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71/output/case_bank
