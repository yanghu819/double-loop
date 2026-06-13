# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.5035, total=0.6647, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, scratch=gated, scratch_delta=5.434, scratch_gauss=0.0000, loop1_loss=1.2699, loop_last_loss=0.5035, sec=8543.3
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3231
  future_seed: fs_gate=0.637, fs_update=1.000, fs_state_norm=20.375, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.233, scratch_decay=0.426, scratch_delta=34.473, scratch_resid=13.454
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4507
  future_seed: fs_gate=0.637, fs_update=1.000, fs_state_norm=20.375, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.240, scratch_decay=0.271, scratch_delta=23.700, scratch_resid=13.356
- loop 3: exact=0.0059, valid=0.0059, solved=0.0059, blank_acc=0.5130
  future_seed: fs_gate=0.637, fs_update=1.000, fs_state_norm=20.375, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.243, scratch_decay=0.243, scratch_delta=15.711, scratch_resid=13.267
- loop 4: exact=0.0059, valid=0.0078, solved=0.0078, blank_acc=0.5237
  future_seed: fs_gate=0.637, fs_update=1.000, fs_state_norm=20.375, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.244, scratch_decay=0.238, scratch_delta=10.234, scratch_resid=13.245
- loop 5: exact=0.0059, valid=0.0098, solved=0.0098, blank_acc=0.5264
  future_seed: fs_gate=0.637, fs_update=1.000, fs_state_norm=20.375, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.245, scratch_decay=0.238, scratch_delta=7.465, scratch_resid=13.242
- loop 6: exact=0.0059, valid=0.0117, solved=0.0117, blank_acc=0.5265
  future_seed: fs_gate=0.637, fs_update=1.000, fs_state_norm=20.375, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, scratch_gate=0.245, scratch_decay=0.237, scratch_delta=5.787, scratch_resid=13.241

## Stochastic Rollouts

- K1: oracle_exact=0.0059, oracle_solved=0.0117, disagree=0.0000
  confidence: exact=0.0059, valid=0.0117, solved=0.0117, blank_acc=0.5265; gap=0.0000
  consistency: exact=0.0059, valid=0.0117, solved=0.0117, blank_acc=0.5265; gap=0.0000
  residual: exact=0.0059, valid=0.0117, solved=0.0117, blank_acc=0.5265; gap=0.0000
  majority: exact=0.0059, valid=0.0117, solved=0.0117, blank_acc=0.5265; gap=0.0000

## Hole Transfer

- holes120: exact=0.0059, valid=0.0117, solved=0.0117, blank_acc=0.5265; indep=0.0000, exact/indep=15742727188219944206614988324864.00
- holes96: exact=0.9883, valid=0.9922, solved=0.9922, blank_acc=0.9987; indep=0.8842, exact/indep=1.12
- holes108: exact=0.8438, valid=0.8496, solved=0.8496, blank_acc=0.9835; indep=0.1663, exact/indep=5.07
- holes132: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1516; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/scratchpad-code-20260612T1012Z-b03829b/runs/scratchpad-gated-h120-step6800-d256-l12-loop6-20260613T0238Z-b03829b/output/futureseed_loop_case_seed52.html
