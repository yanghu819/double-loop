# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.5995, total=1.5995, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.5971, loop_last_loss=1.5995, sec=182.5
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2319, early=0.2335, late=0.2418, early_late_gap=-0.0083
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2319, early=0.2394, late=0.2388, early_late_gap=+0.0005
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Hole Transfer

- holes16: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2319, early=0.2394, late=0.2388, early_late_gap=+0.0005; indep=0.0000, exact/indep=0.00
- holes8: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2588, early=0.2297, late=0.2630, early_late_gap=-0.0333; indep=0.0000, exact/indep=0.00
- holes24: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1934, early=0.1872, late=0.2059, early_late_gap=-0.0187; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-fla-9336e94-20260625T1708/runs/gdn-naive-nofs-small-20260625T1024-5f8a207/output/futureseed_loop_case_seed52.html
