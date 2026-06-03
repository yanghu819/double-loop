# 6x6 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 6x6, box: 2x3, hole pattern: `unit`.

## Loop Metrics

- train: ce=1.9751, total=1.9751, loop_loss=final, noise=feature_diff, buffer=864, loop1_loss=1.9733, loop_last_loss=1.9751, sec=2.8
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000
  future_seed: fs_gate=0.500, fs_state_norm=4.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000
  future_seed: fs_gate=0.500, fs_state_norm=4.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000
  future_seed: fs_gate=0.500, fs_state_norm=4.000

## Eval Noise

- clean eval loop3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000
- noisy eval loop3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000; gap=0.0000

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/smoke-20260603T061813Z-36a87e0/output/futureseed_loop_case_seed52.html
