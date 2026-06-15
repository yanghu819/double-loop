# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.2292, total=0.4431, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.2138, loop_last_loss=0.2292, sec=1895.7
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3797
  future_seed: fs_gate=0.610, fs_update=1.000, fs_state_norm=19.530, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6073
  future_seed: fs_gate=0.610, fs_update=1.000, fs_state_norm=19.530, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0820, valid=0.0820, solved=0.0820, blank_acc=0.7294
  future_seed: fs_gate=0.610, fs_update=1.000, fs_state_norm=19.530, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.1562, valid=0.1602, solved=0.1602, blank_acc=0.7577
  future_seed: fs_gate=0.610, fs_update=1.000, fs_state_norm=19.530, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.1699, valid=0.1719, solved=0.1719, blank_acc=0.7604
  future_seed: fs_gate=0.610, fs_update=1.000, fs_state_norm=19.530, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 6: exact=0.1738, valid=0.1758, solved=0.1758, blank_acc=0.7611
  future_seed: fs_gate=0.610, fs_update=1.000, fs_state_norm=19.530, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.1738, oracle_solved=0.1758, disagree=0.0000
  confidence: exact=0.1738, valid=0.1758, solved=0.1758, blank_acc=0.7611; gap=0.0000
  consistency: exact=0.1738, valid=0.1758, solved=0.1758, blank_acc=0.7611; gap=0.0000
  residual: exact=0.1738, valid=0.1758, solved=0.1758, blank_acc=0.7611; gap=0.0000
  majority: exact=0.1738, valid=0.1758, solved=0.1758, blank_acc=0.7611; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0820, confidence=0.0820, residual=0.0820, disagree=0.0000
### loop4
- K1: oracle_exact=0.1562, confidence=0.1562, residual=0.1562, disagree=0.0000
### loop5
- K1: oracle_exact=0.1699, confidence=0.1699, residual=0.1699, disagree=0.0000
### loop6
- K1: oracle_exact=0.1738, confidence=0.1738, residual=0.1738, disagree=0.0000

## Hole Transfer

- holes120: exact=0.1738, valid=0.1758, solved=0.1758, blank_acc=0.7611; indep=0.0000, exact/indep=29165205119297.11
- holes96: exact=0.9941, valid=0.9941, solved=0.9941, blank_acc=0.9992; indep=0.9303, exact/indep=1.07
- holes108: exact=0.9492, valid=0.9512, solved=0.9512, blank_acc=0.9942; indep=0.5332, exact/indep=1.78
- holes132: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1648; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/d320-mb48eff96-h120-s13600-r2-20260615T210332Z-785f3cd/runs/d320-mb48eff96-h120-s13600-r2-20260615T210332Z-785f3cd/output/futureseed_loop_case_seed52.html
