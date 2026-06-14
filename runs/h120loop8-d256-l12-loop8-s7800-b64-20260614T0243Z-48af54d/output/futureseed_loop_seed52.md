# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.4405, total=0.5916, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.3535, loop_last_loss=0.4405, sec=11824.4
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3298
  future_seed: fs_gate=0.635, fs_update=1.000, fs_state_norm=20.323, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4907
  future_seed: fs_gate=0.635, fs_update=1.000, fs_state_norm=20.323, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.5810
  future_seed: fs_gate=0.635, fs_update=1.000, fs_state_norm=20.323, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0137, valid=0.0137, solved=0.0137, blank_acc=0.5974
  future_seed: fs_gate=0.635, fs_update=1.000, fs_state_norm=20.323, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6012
  future_seed: fs_gate=0.635, fs_update=1.000, fs_state_norm=20.323, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6018
  future_seed: fs_gate=0.635, fs_update=1.000, fs_state_norm=20.323, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 7: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6021
  future_seed: fs_gate=0.635, fs_update=1.000, fs_state_norm=20.323, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 8: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6021
  future_seed: fs_gate=0.635, fs_update=1.000, fs_state_norm=20.323, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0156, oracle_solved=0.0156, disagree=0.0000
  confidence: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6021; gap=0.0000
  consistency: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6021; gap=0.0000
  residual: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6021; gap=0.0000
  majority: exact=0.0156, valid=0.0156, solved=0.0156, blank_acc=0.6021; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0078, confidence=0.0078, residual=0.0078, disagree=0.0000
### loop6
- K1: oracle_exact=0.0156, confidence=0.0156, residual=0.0156, disagree=0.0000
### loop8
- K1: oracle_exact=0.0156, confidence=0.0156, residual=0.0156, disagree=0.0000

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/h120-loop8-scale-code-20260614T0008Z-48af54d/runs/h120loop8-d256-l12-loop8-s7800-b64-20260614T0243Z-48af54d/output/futureseed_loop_case_seed52.html
