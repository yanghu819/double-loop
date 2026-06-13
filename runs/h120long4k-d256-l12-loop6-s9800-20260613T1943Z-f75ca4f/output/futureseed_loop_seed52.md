# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.2030, total=0.3656, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.0086, loop_last_loss=0.2030, sec=11824.9
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3800
  future_seed: fs_gate=0.665, fs_update=1.000, fs_state_norm=21.286, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5534
  future_seed: fs_gate=0.665, fs_update=1.000, fs_state_norm=21.286, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0391, valid=0.0410, solved=0.0410, blank_acc=0.6515
  future_seed: fs_gate=0.665, fs_update=1.000, fs_state_norm=21.286, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0664, valid=0.0703, solved=0.0703, blank_acc=0.6680
  future_seed: fs_gate=0.665, fs_update=1.000, fs_state_norm=21.286, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0723, valid=0.0801, solved=0.0801, blank_acc=0.6699
  future_seed: fs_gate=0.665, fs_update=1.000, fs_state_norm=21.286, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.0723, valid=0.0801, solved=0.0801, blank_acc=0.6704
  future_seed: fs_gate=0.665, fs_update=1.000, fs_state_norm=21.286, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0723, oracle_solved=0.0801, disagree=0.0000
  confidence: exact=0.0723, valid=0.0801, solved=0.0801, blank_acc=0.6704; gap=0.0000
  consistency: exact=0.0723, valid=0.0801, solved=0.0801, blank_acc=0.6704; gap=0.0000
  residual: exact=0.0723, valid=0.0801, solved=0.0801, blank_acc=0.6704; gap=0.0000
  majority: exact=0.0723, valid=0.0801, solved=0.0801, blank_acc=0.6704; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0391, confidence=0.0391, residual=0.0391, disagree=0.0000
### loop6
- K1: oracle_exact=0.0723, confidence=0.0723, residual=0.0723, disagree=0.0000

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/h120longcurve-d256-code-20260613T1755Z-f75ca4f/runs/h120long4k-d256-l12-loop6-s9800-20260613T1943Z-f75ca4f/output/futureseed_loop_case_seed52.html
