# P-GDN3-020: Bounded Log-SPD Native Address Metric

## 1. Metainfo

- Status: approved; implementation complete, strict CUDA contract pending
- Date: 2026-08-12
- First decision field: directional MQAR L1024 from scratch
- Fixed model: D128/L2/H4/K32 pinned-official GDN2 plus native FutureSeed
- Fixed data/training: four KV pairs, 10,000 train and 1,000 validation
  examples, 10 epochs, batch32, seed123
- New parameters: exactly 4,216
- New recurrent state, tokens, scans, custom recurrent kernels, or task rules: zero

## 2. Evidence And Hypothesis

The frozen historical A100 L1024 endpoint reaches balanced accuracy `0.7475`
and joint exact `0.339`. Among its 1,010 wrong query predictions, 815
(`0.806931`) are values that belong to another valid key in the same example.
At L64 all 19 errors are the same wrong-key/valid-value class. This is a
length-dependent address-binding failure rather than absent payload knowledge.

The zero-parameter `P-BIND-001` Sudoku capture independently finds effective
key-rank fraction `0.479188`, anisotropy `6.456761`, condition `402.91`, and
coherence `0.924599`. Its surprise signal does not predict correction
(`AUROC=0.522970`, lift `1.052392`), so it does not authorize a surprise cache.
The A800 reconstruction failed the carrier floor, so its MQAR internal records
cannot formally select a branch.

`P-GDN3-020` therefore tests one independent, carrier-gated prediction: if
poor key geometry causes wrong-key swaps, a learnable bounded Q/K metric should
make the normalized address cloud more isotropic and improve binding without
adding memory slots or task-specific logic.

## 3. Mechanism

After each official GDN2 Q/K ShortConv and SiLU, and before the unchanged
official recurrence, apply the same per-head SPD matrix `C` to Q and K. Write a
symmetric trace-free generator `G`, bound it as `A=G/(1+||G||_F)`, and use
`C=exp(0.5 log(2) A)`. The recurrence still performs its own Q/K normalization.

The actual address metric is `M=C^T C`: it is volume preserving, has FP32
eigenvalues in `[0.5,2]`, and condition below four. Initialization uses
`x + x(C-I)`, so zero raw parameters give exact parent output/state while
retaining a first-order gradient. The contract and endpoint additionally check
the BF16-applied `M`, not only the ideal FP32 factor. Each H4/K32 layer adds
`4*(32*33/2-1)=2,108` parameters; two layers add 4,216.

## 4. Novel Boundary

This is not HYPIC/PIC serving reuse, a FutureSeed producer codec, a surprise
cache, a Raven write controller, V-lifetime control, a parallel expert, or a
precondition residual around the existing state. It changes the learned metric
used by every native read/write address inside the one official GDN2 scan.

It is also narrower than a new rank-2 or clustered recurrence. The experiment
first asks whether address geometry alone is sufficient on the validated
wrong-key regime; Sudoku is not used for this first judgment.

## 5. Integrity Contract

The exact pushed SHA in a clean detached worktree on one visible task-mode GPU
must prove:

1. exact GPU name/UUID, pinned FLA SHA `9c8e42e...d85e`, Zoology SHA
   `1ad20d1`, Triton ShortConv, and two `ChunkGDN2FunctionBackward` paths;
2. control and candidate parent parameter names, tensors, outputs, nonzero-
   incoming-state outputs/states, and parent gradients are exactly equal at
   zero initialization;
3. exact parameter delta 4,216, finite nonzero metric gradient in both layers,
   mechanism dependency, and head-permutation equivariance;
4. opened FP32 `M=C^T C` obeys the analytic bounds and the BF16-applied metric
   remains inside the preregistered `[0.45,2.05]`, condition `<4.60`, absolute
   log-determinant `<=0.10` production bounds;
5. train/test hashes remain `647c64...9a68` / `4a8237...78f`, with no fallback
   or concurrent GPU model/eval; a transparent recorder reproduces control
   logits exactly while seeing both official chunk calls.

Any miss closes this implementation before formal training. No CPU model smoke
is allowed.

## 6. Carrier Admission

The historical run saved predictions but no checkpoint. A formal resource must
therefore train the unchanged runtime control first. It is admitted only if
balanced accuracy is at least `0.70`, joint exact at least `0.25`, and both
past/future accuracy at least `0.68`, with exact source/data/model geometry.

After carrier admission, the same runtime control alone supplies a zero-parameter
128-example address-geometry branch selector. Log-SPD opens only if global
median effective-rank fraction is `<=0.50` or anisotropy is `>=4.0`, and at
least `75%` of layer/head/example records satisfy one of those conditions. The
candidate process is not created until both carrier and branch checks pass. The known
A800 runtime that scored `0.3245/0.0010` is excluded from a repeated formal
attempt. A resource admission failure is not a GDN3 quality result and receives
no seed, epoch, LR, loss, width, depth, or duration rescue.

## 7. Registered Decision Gate

All conditions must pass:

- activation/stability: both layers have actual `||M-I||_F >= 1e-4`; FP32 `M`
  stays in `[0.5,2]` with condition below four and absolute log-determinant
  `<=1e-4`; BF16-applied `M` stays within the fixed production bounds above;
- geometry: on the same fixed 128-example prefix, each layer raises median
  effective-rank fraction by at least `0.05`, reduces median anisotropy by at
  least `20%`, and does not regress future or past own-key binding contrast by
  more than `5%` relative to the matched control;
- state stability: the receiving FutureSeed terminal RMS remains finite,
  nonzero, and at most `2x` the same-runtime control;
- quality: balanced accuracy at least `0.85` and at least `+0.10` over both the
  historical `0.7475` and same-runtime control; joint exact at least `0.60`;
  both directions at least `0.85`;
- binding: wrong-key valid-value swap fraction falls by at least `0.10` versus
  both historical and same-runtime control;
- cost: trainer fit elapsed, complete arm wall time through checkpoint, and
  independent warmed-step elapsed each below `+15%`; peak training allocation
  below `+10%`. Geometry collection is matched and outside the warmed benchmark.

Any miss discards this mechanism. There is no metric cap, parameterization,
seed, LR, loss, batch, width/depth, duration, or nearby conditioner rescue.

## 8. Commands And Artifacts

`scripts/run_zoology_gdn2_log_spd.sh` runs the strict contract, matched control
admission, and only then the candidate. It requires runtime-provided exact
`EXPECTED_GPU_NAME`, `EXPECTED_GPU_UUID`, and pushed `EXPECTED_SOURCE_SHA`.
Both admitted arms save final model state, metrics, cases, timing, memory, and
SHA256 provenance under `/huyang2/double-loop/runs`.

Resource allocation may race task-mode A10080/H80080/A10040 requests, but only the
first exact single-GPU task satisfying this protocol is used; all other pending
requests are closed. The known failing A800 runtime is not rerun.

## 9. Decision

Pending strict contract and a valid same-runtime carrier. A pass would justify
moving the bounded native address metric to the Sudoku D256/L12 scaffold. A
quality or geometry miss closes this static conditioner and redirects the next
architecture to a genuinely new scalable recurrence rather than another
wrapper.
