# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.6116, total=1.6116, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.6077, loop_last_loss=1.6116, sec=69.1
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2417, early=0.2434, late=0.2571, early_late_gap=-0.0137
  future_seed: fs_gate=0.471, fs_update=1.000, fs_state_norm=7.541, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2385, early=0.2383, late=0.2571, early_late_gap=-0.0187
  future_seed: fs_gate=0.471, fs_update=1.000, fs_state_norm=7.541, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Hole Transfer

- holes16: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2385, early=0.2383, late=0.2571, early_late_gap=-0.0187; indep=0.0000, exact/indep=0.00
- holes8: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2681, early=0.2482, late=0.2825, early_late_gap=-0.0343; indep=0.0000, exact/indep=0.00
- holes24: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2010, early=0.1971, late=0.2100, early_late_gap=-0.0128; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-triton-fs-0f5771c-20260625T2035/runs/gdn-triton-fs-s100-20260625T2035-0f5771c/output/futureseed_loop_case_seed52.html
