# 6x6 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 6x6, box: 2x3, hole pattern: `unit`.

## Loop Metrics

- train: ce=2.2045, total=2.2045, loop_loss=final, noise=feature_diff, buffer=1296, loop1_loss=2.2634, loop_last_loss=2.2045, sec=7.5
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1875
  future_seed: fs_gate=0.500, fs_state_norm=6.001
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1875
  future_seed: fs_gate=0.500, fs_state_norm=6.001
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1875
  future_seed: fs_gate=0.500, fs_state_norm=6.001

## Eval Noise

- clean eval loop3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1875
- noisy eval loop3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1250

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1250; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1250; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1250; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.1250; gap=0.0000

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/smoke-20260603T063544Z-eaf6b40/output/futureseed_loop_case_seed52.html
