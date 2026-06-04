# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.0057, total=0.0057, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.0725, loop_last_loss=0.0057, sec=840.9
- loop 1: exact=0.9102, valid=0.9102, solved=0.9102, blank_acc=0.9938
  future_seed: fs_gate=0.524, fs_state_norm=8.380
- loop 2: exact=0.9941, valid=0.9941, solved=0.9941, blank_acc=0.9991
  future_seed: fs_gate=0.524, fs_state_norm=8.380
- loop 3: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994
  future_seed: fs_gate=0.524, fs_state_norm=8.380
- loop 4: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994
  future_seed: fs_gate=0.524, fs_state_norm=8.380
- loop 5: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994
  future_seed: fs_gate=0.524, fs_state_norm=8.380

## Stochastic Rollouts

- K1: oracle_exact=0.9961, oracle_solved=0.9961, disagree=0.0000
  confidence: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994; gap=0.0000
  consistency: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994; gap=0.0000
  residual: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994; gap=0.0000
  majority: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994; gap=0.0000
- K4: oracle_exact=0.9961, oracle_solved=0.9961, disagree=0.0000
  confidence: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994; gap=0.0000
  consistency: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994; gap=0.0000
  residual: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994; gap=0.0000
  majority: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994; gap=0.0000
- K8: oracle_exact=0.9961, oracle_solved=0.9961, disagree=0.0000
  confidence: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994; gap=0.0000
  consistency: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994; gap=0.0000
  residual: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994; gap=0.0000
  majority: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994; gap=0.0000

## Compute Scaling

### loop1
- K1: oracle_exact=0.9102, confidence=0.9102, residual=0.9102, disagree=0.0000
- K4: oracle_exact=0.9102, confidence=0.9102, residual=0.9102, disagree=0.0000
- K8: oracle_exact=0.9102, confidence=0.9102, residual=0.9102, disagree=0.0000
### loop3
- K1: oracle_exact=0.9961, confidence=0.9961, residual=0.9961, disagree=0.0000
- K4: oracle_exact=0.9961, confidence=0.9961, residual=0.9961, disagree=0.0000
- K8: oracle_exact=0.9961, confidence=0.9961, residual=0.9961, disagree=0.0000
### loop5
- K1: oracle_exact=0.9961, confidence=0.9961, residual=0.9961, disagree=0.0000
- K4: oracle_exact=0.9961, confidence=0.9961, residual=0.9961, disagree=0.0000
- K8: oracle_exact=0.9961, confidence=0.9961, residual=0.9961, disagree=0.0000

## Hole Transfer

- holes16: exact=0.9961, valid=0.9961, solved=0.9961, blank_acc=0.9994; indep=0.9903, exact/indep=1.01
- holes20: exact=0.9805, valid=0.9805, solved=0.9805, blank_acc=0.9983; indep=0.9673, exact/indep=1.01
- holes24: exact=0.8828, valid=0.8848, solved=0.8848, blank_acc=0.9897; indep=0.7793, exact/indep=1.13
- holes28: exact=0.6855, valid=0.6914, solved=0.6914, blank_acc=0.9739; indep=0.4770, exact/indep=1.44
- holes32: exact=0.4043, valid=0.4121, solved=0.4121, blank_acc=0.9409; indep=0.1424, exact/indep=2.84

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/rwkv-fs-backbone-cliff2-20260604T070945Z-c9c212a-rwkv_fs-s52/output/futureseed_loop_case_seed52.html
