# 16x16 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 16x16, box: 4x4, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0498, total=1.0510, loop_loss=all, noise=feature_diff, buffer=8192, loop1_loss=1.0595, loop_last_loss=1.0498, sec=2287.7
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3806
  future_seed: fs_gate=0.496, fs_state_norm=7.941
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3857
  future_seed: fs_gate=0.496, fs_state_norm=7.941
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3868
  future_seed: fs_gate=0.496, fs_state_norm=7.941
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3877
  future_seed: fs_gate=0.496, fs_state_norm=7.941
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3873
  future_seed: fs_gate=0.496, fs_state_norm=7.941
- loop 6: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3873
  future_seed: fs_gate=0.496, fs_state_norm=7.941
- loop 7: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3872
  future_seed: fs_gate=0.496, fs_state_norm=7.941
- loop 8: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3872
  future_seed: fs_gate=0.496, fs_state_norm=7.941

## Stochastic Rollouts

- K1: oracle_exact=0.0000, oracle_solved=0.0000, disagree=0.0000
  confidence: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3872; gap=0.0000
  consistency: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3872; gap=0.0000
  residual: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3872; gap=0.0000
  majority: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3872; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop5
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop8
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000

## Hole Transfer

- holes32: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3872; indep=0.0000, exact/indep=0.00
- holes24: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4102; indep=0.0000, exact/indep=0.00
- holes40: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3955; indep=0.0000, exact/indep=0.00
- holes48: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3560; indep=0.0000, exact/indep=0.00

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/frontier-16x16-allloop-d256-l8-20260604T1234Z-75a8796/output/futureseed_loop_case_seed52.html
