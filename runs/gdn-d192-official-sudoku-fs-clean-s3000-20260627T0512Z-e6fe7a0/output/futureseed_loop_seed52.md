# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9633, total=0.9633, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.1972, loop_last_loss=0.9633, sec=1569.4
- loop 1: exact=0.0044, valid=0.0044, solved=0.0044, blank_acc=0.4644, early=0.4681, late=0.4644, early_late_gap=+0.0037
  future_seed: fs_gate=0.480, fs_update=1.000, fs_state_norm=7.687, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0283, valid=0.0283, solved=0.0283, blank_acc=0.5133, early=0.5120, late=0.5214, early_late_gap=-0.0094
  future_seed: fs_gate=0.480, fs_update=1.000, fs_state_norm=7.687, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0288, valid=0.0288, solved=0.0288, blank_acc=0.5385, early=0.5372, late=0.5418, early_late_gap=-0.0046
  future_seed: fs_gate=0.480, fs_update=1.000, fs_state_norm=7.687, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5410, early=0.5393, late=0.5446, early_late_gap=-0.0053
  future_seed: fs_gate=0.480, fs_update=1.000, fs_state_norm=7.687, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0298, valid=0.0298, solved=0.0298, blank_acc=0.5418, early=0.5402, late=0.5454, early_late_gap=-0.0052
  future_seed: fs_gate=0.480, fs_update=1.000, fs_state_norm=7.687, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-clean-s3000-e6fe7a0-20260627/runs/gdn-d192-official-sudoku-fs-clean-s3000-20260627T0512Z-e6fe7a0/output/futureseed_loop_case_seed52.html
