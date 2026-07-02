# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.7241, total=0.8031, loop_loss=all, noise=feature_diff, buffer=8192, dtype=float32, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0011, loop_last_loss=0.7241, sec=305.2
- loop 1: exact=0.0234, valid=0.0234, solved=0.0234, blank_acc=0.5446, early=0.5361, late=0.5506, early_late_gap=-0.0145
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.785, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0312, valid=0.0312, solved=0.0312, blank_acc=0.6061, early=0.5942, late=0.6101, early_late_gap=-0.0159
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.785, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1250, valid=0.1250, solved=0.1250, blank_acc=0.6344, early=0.6289, late=0.6356, early_late_gap=-0.0067
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.785, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.1777, valid=0.1777, solved=0.1777, blank_acc=0.6438, early=0.6377, late=0.6441, early_late_gap=-0.0064
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.785, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.1855, valid=0.1855, solved=0.1855, blank_acc=0.6460, early=0.6390, late=0.6472, early_late_gap=-0.0082
  future_seed: fs_gate=0.462, fs_update=1.000, fs_state_norm=14.785, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369/runs/gdn-transition-mixedhard-d224l12-step6100-20260702T143834Z-0151369/output/futureseed_loop_case_seed52.html
