# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9902, total=1.1556, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.8269, exact_softmin=-0.129, loop1_loss=1.1683, loop_last_loss=0.9902, sec=1745.1
- loop 1: exact=0.0020, valid=0.0020, solved=0.0020, blank_acc=0.4436, early=0.4436, late=0.4370, early_late_gap=+0.0066
  future_seed: fs_gate=0.479, fs_update=1.000, fs_state_norm=7.671, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0273, valid=0.0273, solved=0.0273, blank_acc=0.5022, early=0.4972, late=0.5069, early_late_gap=-0.0097
  future_seed: fs_gate=0.479, fs_update=1.000, fs_state_norm=7.671, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5312, early=0.5293, late=0.5343, early_late_gap=-0.0050
  future_seed: fs_gate=0.479, fs_update=1.000, fs_state_norm=7.671, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5358, early=0.5346, late=0.5387, early_late_gap=-0.0041
  future_seed: fs_gate=0.479, fs_update=1.000, fs_state_norm=7.671, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5362, early=0.5348, late=0.5388, early_late_gap=-0.0039
  future_seed: fs_gate=0.479, fs_update=1.000, fs_state_norm=7.671, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-exactmargin-s3000-6dde82a-20260629/runs/gdn-d192-official-sudoku-fs-exactmargin-s3000-20260629T0251Z-6dde82a/output/futureseed_loop_case_seed52.html
