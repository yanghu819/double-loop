# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.1112, total=0.2134, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=0.6471, loop_last_loss=0.1112, sec=6490.9
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4644
  future_seed: fs_gate=0.620, fs_update=1.000, fs_state_norm=19.852, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.7425
  future_seed: fs_gate=0.620, fs_update=1.000, fs_state_norm=19.852, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.1914, valid=0.1914, solved=0.1914, blank_acc=0.8270
  future_seed: fs_gate=0.620, fs_update=1.000, fs_state_norm=19.852, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.2656, valid=0.2656, solved=0.2656, blank_acc=0.8375
  future_seed: fs_gate=0.620, fs_update=1.000, fs_state_norm=19.852, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.2715, valid=0.2715, solved=0.2715, blank_acc=0.8393
  future_seed: fs_gate=0.620, fs_update=1.000, fs_state_norm=19.852, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.2754, valid=0.2754, solved=0.2754, blank_acc=0.8397
  future_seed: fs_gate=0.620, fs_update=1.000, fs_state_norm=19.852, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.2754, oracle_solved=0.2754, disagree=0.0000
  confidence: exact=0.2754, valid=0.2754, solved=0.2754, blank_acc=0.8397; gap=0.0000
  consistency: exact=0.2754, valid=0.2754, solved=0.2754, blank_acc=0.8397; gap=0.0000
  residual: exact=0.2754, valid=0.2754, solved=0.2754, blank_acc=0.8397; gap=0.0000
  majority: exact=0.2754, valid=0.2754, solved=0.2754, blank_acc=0.8397; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.1914, confidence=0.1914, residual=0.1914, disagree=0.0000
### loop4
- K1: oracle_exact=0.2656, confidence=0.2656, residual=0.2656, disagree=0.0000
### loop5
- K1: oracle_exact=0.2715, confidence=0.2715, residual=0.2715, disagree=0.0000
### loop6
- K1: oracle_exact=0.2754, confidence=0.2754, residual=0.2754, disagree=0.0000

## Hole Transfer

- holes120: exact=0.2754, valid=0.2754, solved=0.2754, blank_acc=0.8397; indep=0.0000, exact/indep=350116073.52
- holes96: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9996; indep=0.9598, exact/indep=1.04
- holes108: exact=0.9492, valid=0.9492, solved=0.9492, blank_acc=0.9944; indep=0.5438, exact/indep=1.75
- holes132: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1693; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/d320-mb48eff96-h120-s22600-20260616T0550Z-1adbda1/runs/d320-mb48eff96-h120-s22600-20260616T0550Z-1adbda1/output/futureseed_loop_case_seed52.html
