# 4x4 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 4x4, box: 2x2, hole pattern: `random`.

## Loop Metrics

- train: ce=1.5523, total=1.5523, loop_loss=final, noise=feature_diff, buffer=64, loop1_loss=1.5523, loop_last_loss=1.5523, sec=14.9
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000
  future_seed: fs_gate=0.500, fs_state_norm=8.000

## Eval Noise

- clean eval loop1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000
- noisy eval loop1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.0000; gap=0.0000

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/rwkv7-statepassing-smoke-20260603T102408Z-3323172/output/futureseed_loop_case_seed52.html
