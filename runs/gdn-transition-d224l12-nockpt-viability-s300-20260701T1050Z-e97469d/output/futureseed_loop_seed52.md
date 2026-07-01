# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0738, total=1.0768, loop_loss=all, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0882, loop_last_loss=1.0738, sec=955.3
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4623, early=0.4522, late=0.4780, early_late_gap=-0.0258
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=15.925, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4708, early=0.4588, late=0.4875, early_late_gap=-0.0287
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=15.925, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4713, early=0.4609, late=0.4864, early_late_gap=-0.0255
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=15.925, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4714, early=0.4613, late=0.4861, early_late_gap=-0.0248
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=15.925, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4717, early=0.4608, late=0.4867, early_late_gap=-0.0259
  future_seed: fs_gate=0.498, fs_update=1.000, fs_state_norm=15.925, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-e97469d-20260701T1050Z/runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-e97469d-20260701T1050Z/runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.0703, blank_acc=0.9179, selected={'solved_by_loop': 2, 'almost_solved': 2, 'hard_failure': 2}
- official_b51_55: index=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-e97469d-20260701T1050Z/runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-e97469d-20260701T1050Z/runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4815, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 2}
- official_b56_64: index=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-e97469d-20260701T1050Z/runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-e97469d-20260701T1050Z/runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4532, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 2}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.0449, valid=0.0449, solved=0.0449, blank_acc=0.9173, early=0.8973, late=0.9491, early_late_gap=-0.0518
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4944, early=0.4840, late=0.5064, early_late_gap=-0.0224
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4429, early=0.4327, late=0.4551, early_late_gap=-0.0224

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-e97469d-20260701T1050Z/runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-e97469d-20260701T1050Z/runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/output/case_bank
