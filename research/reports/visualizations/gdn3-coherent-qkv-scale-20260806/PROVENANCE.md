# P-GDN3-003 Engineering Provenance

- Source commit: `ccd889798ecdfc9e114014699087de7bcc9ac7af`
- Source branch: `codex/gdn3-coherent-init-20260806`
- Remote detached worktree: `/huyang2/double-loop/worktrees/p-gdn3-003-ccd8897`
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Pinned FLA SHA: `9c8e42e762fce087c27b673af4922795d9edb85e`
- CUDA contract SHA256:
  `fd6b1d33e92a3aa159146235a65b33b7ad9610314a455e00d15aefbc418e16fb`
- Two-step fit launch-log SHA256:
  `6ee3d86beea30711718322d8d3f80cfafa6cd751d5eca09a641f5c45ada9d159`
- Remote two-step source-snapshot SHA256:
  `d62a371c6432fedbc937743b231ea5752d1c15442e63a49d94c8d7a4884e6af4`
- Two-step run:
  `gdn3-coherent-qkv-d256l12-fit-20260806T110533Z-ccd8897`
- Formal run:
  `gdn3-coherent-qkv-d256l12-s12000-20260806T111038Z-ccd8897`
- Formal process group/Python at launch: `125922/125990`
- Formal step500 evaluator SHA256:
  `4927b43cfef78ac762bbdfb43413aacc50d5eb1b70e7880bc99bbd1a597755e9`
- Formal abort record SHA256:
  `3716e0d092b74cdc88808a6a4646c45a57361c138b759a2fe4644bc79b40ff1b`
- Final formal launch-log SHA256:
  `3f9095a17292c61208323fd77064e3775b6266be73cd12c9645da25959cb84e8`

The 67 MiB source snapshot remains in the remote fit run directory. Git stores
the exact source commit, source HEAD, contract, configuration, score, logs, and
evaluator/visualization outputs rather than duplicating the generated archive.
The stopped formal log, frozen step500 evaluator JSON, exact abort record,
configuration, and source HEAD are included in this directory.
