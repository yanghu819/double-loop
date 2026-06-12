# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.3685, total=0.5372, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_fb=0.00, loop_time=0.00, loop1_loss=1.1541, loop_last_loss=0.3685, sec=11729.8
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3600
  future_seed: fs_gate=0.659, fs_update=1.000, fs_state_norm=21.094, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5211
  future_seed: fs_gate=0.659, fs_update=1.000, fs_state_norm=21.094, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 3: exact=0.0195, valid=0.0254, solved=0.0254, blank_acc=0.6045
  future_seed: fs_gate=0.659, fs_update=1.000, fs_state_norm=21.094, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 4: exact=0.0371, valid=0.0488, solved=0.0488, blank_acc=0.6181
  future_seed: fs_gate=0.659, fs_update=1.000, fs_state_norm=21.094, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 5: exact=0.0410, valid=0.0527, solved=0.0527, blank_acc=0.6189
  future_seed: fs_gate=0.659, fs_update=1.000, fs_state_norm=21.094, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000
- loop 6: exact=0.0449, valid=0.0586, solved=0.0586, blank_acc=0.6196
  future_seed: fs_gate=0.659, fs_update=1.000, fs_state_norm=21.094, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000

## Stochastic Rollouts

- K1: oracle_exact=0.0449, oracle_solved=0.0586, disagree=0.0000
  confidence: exact=0.0449, valid=0.0586, solved=0.0586, blank_acc=0.6196; gap=0.0000
  consistency: exact=0.0449, valid=0.0586, solved=0.0586, blank_acc=0.6196; gap=0.0000
  residual: exact=0.0449, valid=0.0586, solved=0.0586, blank_acc=0.6196; gap=0.0000
  majority: exact=0.0449, valid=0.0586, solved=0.0586, blank_acc=0.6196; gap=0.0000

## Hole Transfer

- holes120: exact=0.0449, valid=0.0586, solved=0.0586, blank_acc=0.6196; indep=0.0000, exact/indep=398807633850910468210688.00
- holes96: exact=0.9902, valid=0.9922, solved=0.9922, blank_acc=0.9992; indep=0.9284, exact/indep=1.07
- holes108: exact=0.9355, valid=0.9473, solved=0.9473, blank_acc=0.9938; indep=0.5097, exact/indep=1.84
- holes132: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1537; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/h120curve-20260612T0455Z-2e95bd8/runs/h120curve-12x12-d256-l12-loop6-s9000-ckpt1k2k3k-20260612T0455Z-2e95bd8/output/futureseed_loop_case_seed52.html
