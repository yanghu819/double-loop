# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=3.8850, total=3.9042, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=3.9664, loop_last_loss=3.8850, sec=5.2
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0854
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=15.997, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0870
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=15.997, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0849
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=15.997, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0854
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=15.997, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0854
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=15.997, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0854
  future_seed: fs_gate=0.500, fs_update=1.000, fs_state_norm=15.997, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0854; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0854; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0854; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0854; gap=0.0000

## Hole Transfer

- holes120: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0854; indep=0.0000, exact/indep=0.00
- holes96: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0853; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/actckpt-d320-b72-loop6-smoke-20260614T0625Z-c2df534/runs/actckpt-d320-b72-loop6-smoke4-20260614T0622Z-c2df534/output/futureseed_loop_case_seed52.html
