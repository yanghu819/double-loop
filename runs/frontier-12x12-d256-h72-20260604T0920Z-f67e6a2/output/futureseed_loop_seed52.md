# 12x12 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 12x12, box: 3x4, hole pattern: `random`.

## Loop Metrics

- train: ce=0.1857, total=0.1857, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=1.0238, loop_last_loss=0.1857, sec=2111.1
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6990
  future_seed: fs_gate=0.540, fs_state_norm=8.639
- loop 2: exact=0.1615, valid=0.1615, solved=0.1615, blank_acc=0.9471
  future_seed: fs_gate=0.540, fs_state_norm=8.639
- loop 3: exact=0.4427, valid=0.4479, solved=0.4479, blank_acc=0.9717
  future_seed: fs_gate=0.540, fs_state_norm=8.639
- loop 4: exact=0.4844, valid=0.4896, solved=0.4896, blank_acc=0.9738
  future_seed: fs_gate=0.540, fs_state_norm=8.639
- loop 5: exact=0.5052, valid=0.5104, solved=0.5104, blank_acc=0.9743
  future_seed: fs_gate=0.540, fs_state_norm=8.639

## Stochastic Rollouts

- K1: oracle_exact=0.5052, oracle_solved=0.5104, disagree=0.0000
  confidence: exact=0.5052, valid=0.5104, solved=0.5104, blank_acc=0.9743; gap=0.0000
  consistency: exact=0.5052, valid=0.5104, solved=0.5104, blank_acc=0.9743; gap=0.0000
  residual: exact=0.5052, valid=0.5104, solved=0.5104, blank_acc=0.9743; gap=0.0000
  majority: exact=0.5052, valid=0.5104, solved=0.5104, blank_acc=0.9743; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.0000, confidence=0.0000, residual=0.0000, disagree=0.0000
### loop3
- K1: oracle_exact=0.4427, confidence=0.4427, residual=0.4427, disagree=0.0000
### loop5
- K1: oracle_exact=0.5052, confidence=0.5052, residual=0.5052, disagree=0.0000

## Hole Transfer

- holes60: exact=0.5052, valid=0.5104, solved=0.5104, blank_acc=0.9743; indep=0.2103, exact/indep=2.40
- holes48: exact=0.8880, valid=0.8958, solved=0.8958, blank_acc=0.9944; indep=0.7642, exact/indep=1.16
- holes72: exact=0.0443, valid=0.0521, solved=0.0521, blank_acc=0.8881; indep=0.0002, exact/indep=228.23
- holes84: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.6993; indep=0.0000, exact/indep=0.00

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/frontier-12x12-d256-h72-20260604T0920Z-f67e6a2/output/futureseed_loop_case_seed52.html
