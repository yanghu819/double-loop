# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0585, total=0.0585, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.3665, loop_last_loss=0.0585, sec=1255.2
- loop 1: exact=0.3320, valid=0.3320, solved=0.3320, blank_acc=0.9511
  future_seed: fs_gate=0.543, fs_state_norm=8.688
- loop 2: exact=0.9043, valid=0.9082, solved=0.9082, blank_acc=0.9916
  future_seed: fs_gate=0.543, fs_state_norm=8.688
- loop 3: exact=0.9199, valid=0.9238, solved=0.9238, blank_acc=0.9926
  future_seed: fs_gate=0.543, fs_state_norm=8.688
- loop 4: exact=0.9199, valid=0.9238, solved=0.9238, blank_acc=0.9930
  future_seed: fs_gate=0.543, fs_state_norm=8.688
- loop 5: exact=0.9199, valid=0.9238, solved=0.9238, blank_acc=0.9930
  future_seed: fs_gate=0.543, fs_state_norm=8.688

## Stochastic Rollouts

- K1: oracle_exact=0.9199, oracle_solved=0.9238, disagree=0.0000
  confidence: exact=0.9199, valid=0.9238, solved=0.9238, blank_acc=0.9930; gap=0.0000
  consistency: exact=0.9199, valid=0.9238, solved=0.9238, blank_acc=0.9930; gap=0.0000
  residual: exact=0.9199, valid=0.9238, solved=0.9238, blank_acc=0.9930; gap=0.0000
  majority: exact=0.9199, valid=0.9238, solved=0.9238, blank_acc=0.9930; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.3320, confidence=0.3320, residual=0.3320, disagree=0.0000
### loop3
- K1: oracle_exact=0.9199, confidence=0.9199, residual=0.9199, disagree=0.0000
### loop5
- K1: oracle_exact=0.9199, confidence=0.9199, residual=0.9199, disagree=0.0000

## Hole Transfer

- holes32: exact=0.9199, valid=0.9238, solved=0.9238, blank_acc=0.9930; indep=0.7998, exact/indep=1.15
- holes24: exact=0.9883, valid=0.9902, solved=0.9902, blank_acc=0.9980; indep=0.9542, exact/indep=1.04
- holes36: exact=0.8086, valid=0.8398, solved=0.8398, blank_acc=0.9796; indep=0.4762, exact/indep=1.70
- holes40: exact=0.5332, valid=0.5918, solved=0.5918, blank_acc=0.9531; indep=0.1466, exact/indep=3.64
- holes44: exact=0.1914, valid=0.2090, solved=0.2090, blank_acc=0.8893; indep=0.0057, exact/indep=33.34

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/scaleup-d192l10-h40-20260604T0844Z-a5b3b4a/output/futureseed_loop_case_seed52.html
