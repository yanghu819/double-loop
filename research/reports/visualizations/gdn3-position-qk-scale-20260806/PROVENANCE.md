# P-GDN3-004 Provenance

- Source commit: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Source branch: `codex/gdn3-coherent-init-20260806`
- Remote detached worktree: `/huyang2/double-loop/worktrees/p-gdn3-004-9f2ee8d`
- Formal run: `gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d`
- Formal process group/Python child: `128112/128187`
- GPU visible index/UUID: `0` / `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Pinned FLA SHA: `9c8e42e762fce087c27b673af4922795d9edb85e`
- Formal configuration SHA256:
  `351faf45e9e50f9ddf8e281ae867291cd0b9acaecef8d7600c848e451ba67a96`
- Formal step500 evaluator SHA256:
  `58356e226d798e6a761bc945ad90d50b4eea3f903c233b9b551419e18d369369`
- Formal source HEAD record SHA256:
  `55b8078030e8d7ccb6b4f385ec9db26d52a7d6afe9701026364cf49b14f487d2`
- Through-step500 launch-log snapshot SHA256:
  `70777c6bdaa8cf2bbed5d3d42c6fa2a8eeb59201e6e327aa7e9f7b6545478220`
- Remote source snapshot SHA256:
  `e5b3a28602ac29e651b5465b92a8b0645449b0597418a3aaa12de354d35b3fd0`
- Formal step500 checkpoint SHA256:
  `522a0a96950b7d9ef30bd65c7b93cae98265fbd3a7fef7e21a3663d92d4fd602`

The stable post-compile checkpoint intervals from step100 through step400 are
`716/723/725` seconds per 100 steps. Their mean is `7.2133` seconds/step,
equivalent to `17.745` effective boards/second at batch128, with interval CV
`0.535%`. The step400-to-step500 interval was `770` seconds and is retained in
the raw timestamps rather than folded into the stable-window throughput.

The exact full-shape fit peak allocation was `13,045.7 MiB`. A live formal-run
NVML sample during step500 evaluation was `17,491 MiB`; the formal process has
not yet emitted its endpoint `torch.cuda.max_memory_allocated`, so that value
is intentionally left open until the frozen endpoint.

The evaluator JSON and HTML summarize aggregate fixed probes only. They are not
full official blank ranges and do not contain per-board predictions. The live
launch log continues to grow remotely; `formal-through-step500.log` is the
frozen local snapshot used by the hash above.

## Step1000 Diagnostic And Step3000 Stop

- Formal step1000 evaluator SHA256:
  `ffc8d9261f4329d161131d3a3fe10cc65ae5f86e3bb142197bcc2600140e5761`
- Formal step3000 evaluator SHA256:
  `d71520ed1dc00abe14d6b52adae96f2eb58b8c212b818393912e9beced666217`
- Formal step3000 checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Formal abort record SHA256:
  `94d34ea215bfed053f424719891dc750f302ff91dc4f3f146b3c342aa318ddb8`
- Stopped launch-log SHA256:
  `675be197346a46bbe67ec1e874c3c8ea654ec58f419c956b879ac01b08cb06c3`
- Stop time: `2026-08-06T19:32:13Z`
- Stop identity: exact PGID/leader `128112`, Python child `128187`,
  SIGTERM, no escalation
- Registered decision: fail because h53 loop5 exact/blank
  `0.003906/0.582658` is below both `0.02/0.60` alternatives
- Integrity before stop: no NaN, OOM, fallback, source, data, SHA, or GPU drift

Nineteen non-evaluation checkpoint intervals from step1100 through step3000
average `735.785` seconds per 100 steps, equivalent to `17.396`
effective boards/second, with `0.479%` interval CV. Training exposure at
the stop is 384,000 effective boards, 31.104M board input cell tokens, and
155.52M loop-cell evaluations.

The step3000 HTML and summary again contain aggregate fixed probes only. They
are not official blank-range evaluations and contain no per-board
trajectories. The checkpoint remains eligible only as the preregistered
P-FS3-001 parent because h53 exact is nonzero from loop3 onward and h64 has
genuine loop-wise wrong-cell reduction.
