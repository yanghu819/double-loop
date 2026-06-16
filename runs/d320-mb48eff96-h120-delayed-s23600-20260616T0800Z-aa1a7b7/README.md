# D320 Packed H120 Delayed Loop Credit Probe

- Timestamp: 2026-06-16T08:12Z launch, 2026-06-16T08:40:41Z aborted
- Source SHA: `aa1a7b75b0844350849de887160044cdeaf677d8`
- Remote worktree: `/huyang2/double-loop/.worktrees/d320-mb48eff96-h120-delayed-s23600-20260616T0800Z-aa1a7b7`
- Resume checkpoint: packed D320 step22600 `latest.pt`
- GPU: GPU1 only

## Hypothesis

The step22600 run showed that loop4 does most of the h120 work, while loop5/6
add little. One possible cause is objective shape: all-loop supervision may make
every loop learn to stabilize early instead of giving late loops credit for
correction. This probe changes only loop credit, not model structure:
`LOOP_LOSS=delayed`, `LOOP_LOSS_START=4`, `LOOP_LOSS_POWER=2.0`.

## Result

Early checkpoint at step23100:

- h96/h108/h120/h132 loop6 exact: `0.9980` / `0.9355` / `0.1602` / `0.0000`
- h120 loop1/3/4/5/6 exact: `0.0000` / `0.0781` / `0.1406` / `0.1563` / `0.1602`
- h120 loop6 blank_acc: `0.7804`
- train step23100: `ce=0.2087`, delayed `total=0.2094`, `loop1=1.3307`

The stop decision was made after the step23100 checkpoint. The initial parent
PID stop left orphaned `run.sh`/`tee`/Python child processes, which were then
stopped by exact PIDs after the log reached step23200. `abort.json` records the
parent PID, child PIDs, and GPU cleanup.

## Decision

Stop delayed loop credit in this form. It preserves easy-hole transfer, but it
damages the h120 frontier versus the clean step22600 baseline (`0.2754` final,
`0.2461` checkpoint at step22600). This is not a useful way to make late loops
keep correcting boards.

The negative result is informative: simply moving loss weight to later loops
does not create better late-loop reasoning. The next mechanism should change
state dynamics or effective scale, not loop-loss credit.
