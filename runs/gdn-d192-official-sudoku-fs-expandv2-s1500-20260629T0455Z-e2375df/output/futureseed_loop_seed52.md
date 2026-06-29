# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9812, total=0.9812, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.1687, loop_last_loss=0.9812, sec=1722.4
- loop 1: exact=0.0010, valid=0.0010, solved=0.0010, blank_acc=0.4560, early=0.4554, late=0.4622, early_late_gap=-0.0068
  future_seed: fs_gate=0.481, fs_update=1.000, fs_state_norm=10.894, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0254, valid=0.0254, solved=0.0254, blank_acc=0.5110, early=0.5079, late=0.5165, early_late_gap=-0.0087
  future_seed: fs_gate=0.481, fs_update=1.000, fs_state_norm=10.895, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5312, early=0.5292, late=0.5335, early_late_gap=-0.0043
  future_seed: fs_gate=0.481, fs_update=1.000, fs_state_norm=10.895, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5357, early=0.5344, late=0.5363, early_late_gap=-0.0019
  future_seed: fs_gate=0.481, fs_update=1.000, fs_state_norm=10.895, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5364, early=0.5357, late=0.5368, early_late_gap=-0.0011
  future_seed: fs_gate=0.481, fs_update=1.000, fs_state_norm=10.895, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official: index=/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-expandv2-s1500-e2375df-20260629/runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df/output/case_bank/official/index.html; cases=/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-expandv2-s1500-e2375df-20260629/runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df/output/case_bank/official/cases.json; final_loop=5, exact=0.0352, blank_acc=0.5410, selected={'solved_by_loop': 3, 'almost_solved': 0, 'hard_failure': 3}

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-expandv2-s1500-e2375df-20260629/runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-expandv2-s1500-e2375df-20260629/runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df/output/case_bank
