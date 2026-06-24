# official-eqr-bidir-sudoku-scale256-20260624

## 1. Metainfo

- Plan ID: `P-EQR-012`
- Status: completed, truncated after paired step500 metrics
- Machine: AIStation `GPU1` only
- Remote worktree: `/huyang2/double-loop/.worktrees/official-eqr-bidir-probe-45fe80a`
- Source SHA: `45fe80a4b31426b81fae6dda99997871af6c6a4c`
- Run stamp: `20260624T0415Z-bidir256-45fe80a`

## 2. Hypothesis

The quick cheap-bidirectional probe showed a large recurrent residual drop but
no exact solves. If FutureSeed is a cheap way for a causal recurrent backbone to
access future context, a longer matched-compute budget should convert that
residual contraction into earlier exact accuracy or at least a larger state
convergence advantage.

This is not a seed sweep. It is the single scale gate that the quick probe
unlocked.

## 3. Configuration

- Official EqR upstream base: pristine `locuslab/EqR` at
  `aba94e9cde0f273ce644db5261cd6915ba6561f0`, patched by project launcher at
  source SHA `45fe80a`.
- Dataset: official `sudoku-extreme-1k-aug-1000`
- Arms:
  - `causal-cheap`: `arch.attention_causal=true`, `arch.mlp_t=false`
  - `causal-bidir-futureseed`: same cheap causal backbone plus
    `future_seed_mode=reverse_causal`
- Shared overrides:
  - `arch.hidden_size=192`
  - `arch.num_heads=6`
  - `arch.halt_max_steps=16`
  - `arch.noise_scale=0.01`
  - `epochs=256`
  - `global_batch_size=128`
  - `eval_interval_steps=250`
  - `checkpoint_interval_steps=250`

No Sudoku solver, repair, selector, oracle rollout, or task-specific rule is
used.

## 4. Prediction And Kill Criteria

Prediction:

- FutureSeed should preserve or enlarge the residual advantage seen in the
  quick run (`residual_of_16_steps` much lower than causal-cheap).
- A useful result requires either earlier nonzero exact accuracy or clearly
  better convergence without worse loss.

Kill criteria:

- If by the first extended eval both arms still have exact `0` and FutureSeed's
  residual advantage collapses, stop this cheap-bidirectional path.
- If GPU utilization stays low after compile while memory remains high, stop and
  inspect instead of burning time.
- Do not add seeds, gate-bias sweeps, loss-weight sweeps, or hidden-size tables
  from this run.

## 5. Commands

The intended detached remote command used the existing launcher with only budget
overrides:

```bash
RUN_STAMP=20260624T0415Z-bidir256-45fe80a \
LOG_ROOT=/huyang2/double-loop/official_eqr_compare/artifacts/bidir_sudoku_probe_20260624T0415Z-bidir256-45fe80a \
EPOCHS=256 TRAIN_EPOCHS_PER_ITER=256 GLOBAL_BATCH_SIZE=128 \
CUDA_VISIBLE_DEVICES=0 \
bash runs/official-eqr-bidir-sudoku-probe-20260623/launch_probe.sh
```

The first AIStation exec returned a shell-parse failure, but it had already
launched the detached job. A second base64-safe launch attempt was guarded out
because the same run was already active; no duplicate training was started.

## 6. Artifacts

- Launch PID: `/huyang2/double-loop/artifacts/launch/official-eqr-bidir-sudoku-20260624/20260624T0415Z-bidir256-45fe80a.pid`
- Latest long run stamp:
  `/huyang2/double-loop/artifacts/launch/official-eqr-bidir-sudoku-20260624/latest_long_run_stamp.txt`
- Remote log root:
  `/huyang2/double-loop/official_eqr_compare/artifacts/bidir_sudoku_probe_20260624T0415Z-bidir256-45fe80a`
- causal-cheap log:
  `/huyang2/double-loop/official_eqr_compare/logs/official-eqr-causal-cheap-sudoku-20260624T0415Z-bidir256-45fe80a.log`
- causal-bidir-futureseed log:
  `/huyang2/double-loop/official_eqr_compare/logs/official-eqr-causal-bidir-futureseed-sudoku-20260624T0415Z-bidir256-45fe80a.log`

## 7. Live Status

Launched on GPU1. Confirmed active at `2026-06-24T04:12:23Z`:

- launcher pid: `4251`
- launcher shell: `4252`
- active causal-cheap python processes: `4388`, `4460`, `4536`
- GPU: `NVIDIA A100-SXM4-80GB`, `CUDA_VISIBLE_DEVICES=0`

Operator note: the first long command returned a shell-parse failure, but it had
already launched the detached job. The second base64-safe command correctly
guarded out on the active run, so no duplicate job was started.

The causal-cheap arm was stopped after step500 metrics were available, because
further cheap-only eval before seeing FutureSeed had low information gain. The
launcher then entered the FutureSeed arm. The FutureSeed arm was stopped after
step500 paired metrics for the same reason.

## 8. Results

Remote partial comparison:
`/huyang2/double-loop/official_eqr_compare/artifacts/bidir_sudoku_probe_20260624T0415Z-bidir256-45fe80a/comparison_partial_step250_500.json`.

Local archived comparison:
`runs/official-eqr-bidir-sudoku-scale256-20260624/comparison_partial_step250_500.json`.

| step | metric | causal-cheap | causal-bidir-futureseed | FS - cheap |
|---:|---|---:|---:|---:|
| 250 | `accuracy` | `0.0931839` | `0.0995598` | `+0.0063759` |
| 250 | `exact_accuracy` | `0.0` | `0.0` | `0.0` |
| 250 | `lm_loss` | `2.5436098` | `2.5615923` | `+0.0179825` |
| 250 | `total_loss` | `2.5477071` | `2.5656831` | `+0.0179760` |
| 250 | `residual_of_16_steps` | `167.7596` | `83.3399` | `-84.4197` |
| 500 | `accuracy` | `0.1133033` | `0.1143073` | `+0.0010040` |
| 500 | `exact_accuracy` | `0.0` | `0.0` | `0.0` |
| 500 | `lm_loss` | `2.4121723` | `2.4129703` | `+0.0007980` |
| 500 | `total_loss` | `2.4161344` | `2.4169447` | `+0.0008103` |
| 500 | `residual_of_16_steps` | `121.4452` | `111.6456` | `-9.7996` |

Both arms were truncated after paired step500 metrics. GPU1 was idle after
truncation.

## 9. Decision

Do not continue this exact run or turn it into a seed/epoch table.

The mechanism signal is real but not yet useful enough for a paper claim:
FutureSeed consistently gives the cheap causal backbone a much lower early
16-step residual, which supports the "cheap future context changes recurrent
state dynamics" story. But by step500 the advantage has mostly collapsed, exact
accuracy remains zero, and loss is still slightly worse. This says FutureSeed
as currently injected helps the trajectory contract early, but it does not
convert that contraction into valid Sudoku solves under this official EqR
training objective.

Next high-ROI move is not more epochs or seeds. We need either:

- a metric/visual probe that explains why lower residual is not becoming exact,
  or
- a simple generic state/readout mechanism that turns residual contraction into
  a decision improvement.

For the paper story, the current result supports only a narrow analysis claim:
FutureSeed can provide cheap future-conditioned state movement in a causal
recurrent backbone. It does not yet show superiority over EqR.
