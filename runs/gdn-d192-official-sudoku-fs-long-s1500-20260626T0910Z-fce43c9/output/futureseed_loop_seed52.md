# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9875, total=0.9875, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.2053, loop_last_loss=0.9875, sec=1003.1
- loop 1: exact=0.0054, valid=0.0054, solved=0.0054, blank_acc=0.4547, early=0.4417, late=0.4571, early_late_gap=-0.0154
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=7.810, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5177, early=0.5153, late=0.5230, early_late_gap=-0.0077
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=7.810, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5323, early=0.5312, late=0.5358, early_late_gap=-0.0046
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=7.810, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5334, early=0.5324, late=0.5362, early_late_gap=-0.0038
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=7.810, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5337, early=0.5330, late=0.5367, early_late_gap=-0.0037
  future_seed: fs_gate=0.488, fs_update=1.000, fs_state_norm=7.810, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-b471906-20260626/runs/gdn-d192-official-sudoku-fs-long-s1500-20260626T0910Z-fce43c9/output/futureseed_loop_case_seed52.html
