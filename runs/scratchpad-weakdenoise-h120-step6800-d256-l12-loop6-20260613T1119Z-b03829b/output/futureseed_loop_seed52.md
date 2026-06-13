# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.3505, total=0.4742, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=gated, scratch_delta=3.923, scratch_gauss=0.2728, loop1_loss=0.9882, loop_last_loss=0.3505, sec=7935.5
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4361
  future_seed: fs_gate=0.634, fs_update=1.000, fs_state_norm=20.274, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.205, scratch_decay=0.361, scratch_delta=37.709, scratch_resid=13.621, scratch_var=0.418, scratch_rank=61.2
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6163
  future_seed: fs_gate=0.634, fs_update=1.000, fs_state_norm=20.274, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.218, scratch_decay=0.281, scratch_delta=27.368, scratch_resid=13.183, scratch_var=0.715, scratch_rank=62.2
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6641
  future_seed: fs_gate=0.634, fs_update=1.000, fs_state_norm=20.274, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.220, scratch_decay=0.275, scratch_delta=14.498, scratch_resid=13.013, scratch_var=0.801, scratch_rank=62.5
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6711
  future_seed: fs_gate=0.634, fs_update=1.000, fs_state_norm=20.274, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.221, scratch_decay=0.275, scratch_delta=8.494, scratch_resid=12.984, scratch_var=0.816, scratch_rank=62.5
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6720
  future_seed: fs_gate=0.634, fs_update=1.000, fs_state_norm=20.274, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.221, scratch_decay=0.277, scratch_delta=5.699, scratch_resid=12.978, scratch_var=0.820, scratch_rank=62.5
- loop 6: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6723
  future_seed: fs_gate=0.634, fs_update=1.000, fs_state_norm=20.274, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.221, scratch_decay=0.277, scratch_delta=4.174, scratch_resid=12.977, scratch_var=0.820, scratch_rank=62.5

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6723; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6723; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6723; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6723; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop6
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/scratchpad-weakdenoise-code-20260613T1118Z-b03829b/runs/scratchpad-weakdenoise-h120-step6800-d256-l12-loop6-20260613T1119Z-b03829b/output/futureseed_loop_case_seed52.html
