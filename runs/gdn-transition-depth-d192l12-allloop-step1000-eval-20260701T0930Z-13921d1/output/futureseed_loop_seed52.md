# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9291, total=0.9362, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=0.9602, loop_last_loss=0.9291, sec=0.5
- loop 1: exact=0.0283, valid=0.0283, solved=0.0283, blank_acc=0.5100, early=0.5150, late=0.5093, early_late_gap=+0.0057
  future_seed: fs_gate=0.494, fs_update=1.000, fs_state_norm=15.804, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0283, valid=0.0283, solved=0.0283, blank_acc=0.5243, early=0.5282, late=0.5258, early_late_gap=+0.0024
  future_seed: fs_gate=0.494, fs_update=1.000, fs_state_norm=15.804, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0283, valid=0.0283, solved=0.0283, blank_acc=0.5244, early=0.5286, late=0.5252, early_late_gap=+0.0034
  future_seed: fs_gate=0.494, fs_update=1.000, fs_state_norm=15.804, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0283, valid=0.0283, solved=0.0283, blank_acc=0.5249, early=0.5283, late=0.5260, early_late_gap=+0.0023
  future_seed: fs_gate=0.494, fs_update=1.000, fs_state_norm=15.804, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0283, valid=0.0283, solved=0.0283, blank_acc=0.5245, early=0.5281, late=0.5258, early_late_gap=+0.0023
  future_seed: fs_gate=0.494, fs_update=1.000, fs_state_norm=15.804, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-allloop-13921d1-20260701T0845Z/runs/gdn-transition-depth-d192l12-allloop-step1000-eval-20260701T0930Z-13921d1/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-allloop-13921d1-20260701T0845Z/runs/gdn-transition-depth-d192l12-allloop-step1000-eval-20260701T0930Z-13921d1/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.9922, blank_acc=0.9998, selected={'solved_by_loop': 3, 'almost_solved': 2, 'hard_failure': 0}
- official_b51_55: index=/huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-allloop-13921d1-20260701T0845Z/runs/gdn-transition-depth-d192l12-allloop-step1000-eval-20260701T0930Z-13921d1/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-allloop-13921d1-20260701T0845Z/runs/gdn-transition-depth-d192l12-allloop-step1000-eval-20260701T0930Z-13921d1/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5414, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 3}
- official_b56_64: index=/huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-allloop-13921d1-20260701T0845Z/runs/gdn-transition-depth-d192l12-allloop-step1000-eval-20260701T0930Z-13921d1/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-allloop-13921d1-20260701T0845Z/runs/gdn-transition-depth-d192l12-allloop-step1000-eval-20260701T0930Z-13921d1/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4910, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 3}

## Official Blank-Range Eval

- b46_50 (46-50, n=1024): exact=0.9971, valid=0.9971, solved=0.9971, blank_acc=0.9999, early=0.9999, late=0.9999, early_late_gap=+0.0001
- b51_55 (51-55, n=1024): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5447, early=0.5410, late=0.5439, early_late_gap=-0.0029
- b56_64 (56-64, n=1024): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4927, early=0.4908, late=0.4899, early_late_gap=+0.0009

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-allloop-13921d1-20260701T0845Z/runs/gdn-transition-depth-d192l12-allloop-step1000-eval-20260701T0930Z-13921d1/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-allloop-13921d1-20260701T0845Z/runs/gdn-transition-depth-d192l12-allloop-step1000-eval-20260701T0930Z-13921d1/output/case_bank
