# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0667, total=1.0667, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.2253, loop_last_loss=1.0667, sec=212.1
- loop 1: exact=0.0010, valid=0.0010, solved=0.0010, blank_acc=0.4444, early=0.4555, late=0.4288, early_late_gap=+0.0267
  future_seed: fs_gate=0.476, fs_update=1.000, fs_state_norm=7.617, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.4910, early=0.4904, late=0.4908, early_late_gap=-0.0004
  future_seed: fs_gate=0.476, fs_update=1.000, fs_state_norm=7.617, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0107, valid=0.0107, solved=0.0107, blank_acc=0.4927, early=0.4931, late=0.4914, early_late_gap=+0.0017
  future_seed: fs_gate=0.476, fs_update=1.000, fs_state_norm=7.617, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0107, valid=0.0107, solved=0.0107, blank_acc=0.4925, early=0.4925, late=0.4913, early_late_gap=+0.0012
  future_seed: fs_gate=0.476, fs_update=1.000, fs_state_norm=7.617, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-scale-fs-clean-be2815b-20260626/runs/gdn-realbwd-official-sudoku-scale-fs-clean-20260626T0105-be2815b/output/futureseed_loop_case_seed52.html
