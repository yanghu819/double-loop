# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.5760, total=0.7021, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=gated, scratch_delta=4.334, scratch_gauss=0.1493, loop1_loss=1.1920, loop_last_loss=0.5760, sec=8526.6
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3810
  future_seed: fs_gate=0.637, fs_update=1.000, fs_state_norm=20.373, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.201, scratch_decay=0.418, scratch_delta=37.004, scratch_resid=13.333, scratch_var=0.512, scratch_rank=61.8
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5376
  future_seed: fs_gate=0.637, fs_update=1.000, fs_state_norm=20.373, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.245, scratch_decay=0.271, scratch_delta=26.356, scratch_resid=12.971, scratch_var=0.793, scratch_rank=62.5
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5897
  future_seed: fs_gate=0.637, fs_update=1.000, fs_state_norm=20.373, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.262, scratch_decay=0.250, scratch_delta=16.189, scratch_resid=12.977, scratch_var=0.820, scratch_rank=62.8
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5965
  future_seed: fs_gate=0.637, fs_update=1.000, fs_state_norm=20.373, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.266, scratch_decay=0.248, scratch_delta=9.446, scratch_resid=12.999, scratch_var=0.816, scratch_rank=62.8
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5975
  future_seed: fs_gate=0.637, fs_update=1.000, fs_state_norm=20.373, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.268, scratch_decay=0.248, scratch_delta=6.207, scratch_resid=13.007, scratch_var=0.816, scratch_rank=62.8
- loop 6: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5977
  future_seed: fs_gate=0.637, fs_update=1.000, fs_state_norm=20.373, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.268, scratch_decay=0.247, scratch_delta=4.423, scratch_resid=13.009, scratch_var=0.816, scratch_rank=62.8

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5977; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5977; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5977; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5977; gap=0.0000

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

- case_html: /huyang2/double-loop/.worktrees/scratchpad-noisygauss-code-20260613T0520Z-b03829b/runs/scratchpad-noisygauss-h120-step6800-d256-l12-loop6-20260613T0638Z-b03829b/output/futureseed_loop_case_seed52.html
