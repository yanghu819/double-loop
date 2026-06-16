# D320 Packed H120 Step19600 Continuation

- Timestamp: 2026-06-16T04:00Z launch, 2026-06-16T05:42:59Z recorded
- Source SHA: `7e1ddbaa2bf150946bf13a10b35c8c361e901576`
- Remote worktree: `/huyang2/double-loop/.worktrees/d320-mb48eff96-h120-s19600-20260616T0400Z-7e1ddba`
- Resume checkpoint: packed D320 step16600 `latest.pt`
- GPU: GPU1 only

## Hypothesis

The step16600 run beat the old h120 platform, but h132 stayed closed. This run
tests whether h120 exact still scales with more clean packed compute or whether
the system has entered a local-fill plateau.

The mechanism stays unchanged: fixed FutureSeed+loop, loop6, all-loop loss, CUDA
statepassing RWKV7, no selector, no repair, no scratch, no feature noise.

## Results

Checkpoint h120 loop6 exact:

- step17600: `0.1934`, blank_acc `0.7993`
- step18600: `0.2266`, blank_acc `0.8016`
- step19600: `0.2520`, blank_acc `0.8104`

Final full eval:

- score: `0.2891`
- h96/h108/h120/h132 loop6 exact: `0.9961` / `0.9512` / `0.2891` / `0.0000`
- h120 loop1/3/4/5/6 exact: `0.0000` / `0.2070` / `0.2773` / `0.2852` / `0.2891`
- h120 loop6 blank_acc: `0.8342`
- K1 oracle gap: `0.0000`

## Decision

Continue clean packed D320 scaling. The final h120 exact moves from `0.2363` to
`0.2891`, and the checkpoint curve recovers from the lower step17600 point to a
new step19600 high. This argues against switching yet to selector, repair, or a
new state-update mechanism.

The main limit is still h132 transfer: h132 exact is `0.0` and blank_acc is only
about `0.165`. Treat h120 as the active frontier until it is much more supported;
do not jump to h132 training before h120 is better solved.

Loop remains the solver, but not the obvious next scaling axis: loop1 is zero,
loop3 already solves `0.2070`, and loop5/6 are close. Buy more training, not a
loop-count sweep.

