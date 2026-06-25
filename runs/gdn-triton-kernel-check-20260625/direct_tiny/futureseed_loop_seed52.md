# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=2.4179, total=2.4179, loop_loss=final, noise=feature_diff, buffer=2592, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=2.4157, loop_last_loss=2.4179, sec=3.7
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1250, early=0.1905, late=0.0870, early_late_gap=+0.1035
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=8.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1250, early=0.1905, late=0.0870, early_late_gap=+0.1035
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=8.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/artifacts/gdn_triton_direct_tiny/futureseed_loop_case_seed52.html
