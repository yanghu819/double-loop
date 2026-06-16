# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.1415, total=0.2990, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=0.9348, loop_last_loss=0.1415, sec=6722.3
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4564
  future_seed: fs_gate=0.621, fs_update=1.000, fs_state_norm=19.870, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0137, valid=0.0137, solved=0.0137, blank_acc=0.7228
  future_seed: fs_gate=0.621, fs_update=1.000, fs_state_norm=19.870, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.2070, valid=0.2109, solved=0.2109, blank_acc=0.8171
  future_seed: fs_gate=0.621, fs_update=1.000, fs_state_norm=19.870, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.2773, valid=0.2852, solved=0.2852, blank_acc=0.8312
  future_seed: fs_gate=0.621, fs_update=1.000, fs_state_norm=19.870, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.2852, valid=0.2910, solved=0.2910, blank_acc=0.8338
  future_seed: fs_gate=0.621, fs_update=1.000, fs_state_norm=19.870, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.2891, valid=0.2969, solved=0.2969, blank_acc=0.8342
  future_seed: fs_gate=0.621, fs_update=1.000, fs_state_norm=19.870, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.2891, oracle_solved=0.2969, disagree=0.0000
  confidence: exact=0.2891, valid=0.2969, solved=0.2969, blank_acc=0.8342; gap=0.0000
  consistency: exact=0.2891, valid=0.2969, solved=0.2969, blank_acc=0.8342; gap=0.0000
  residual: exact=0.2891, valid=0.2969, solved=0.2969, blank_acc=0.8342; gap=0.0000
  majority: exact=0.2891, valid=0.2969, solved=0.2969, blank_acc=0.8342; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.2070, confidence=0.2070, residual=0.2070, disagree=0.0000
### loop4
- K1: oracle_exact=0.2773, confidence=0.2773, residual=0.2773, disagree=0.0000
### loop5
- K1: oracle_exact=0.2852, confidence=0.2852, residual=0.2852, disagree=0.0000
### loop6
- K1: oracle_exact=0.2891, confidence=0.2891, residual=0.2891, disagree=0.0000

## Hole Transfer

- holes120: exact=0.2891, valid=0.2969, solved=0.2969, blank_acc=0.8342; indep=0.0000, exact/indep=804954861.09
- holes96: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9996; indep=0.9579, exact/indep=1.04
- holes108: exact=0.9512, valid=0.9512, solved=0.9512, blank_acc=0.9946; indep=0.5601, exact/indep=1.70
- holes132: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1650; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/d320-mb48eff96-h120-s19600-20260616T0400Z-7e1ddba/runs/d320-mb48eff96-h120-s19600-20260616T0400Z-7e1ddba/output/futureseed_loop_case_seed52.html
