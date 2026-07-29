# Flash Linear Attention 9c8e42e Wheel

- Upstream: `https://github.com/fla-org/flash-linear-attention`
- Commit: `9c8e42e762fce087c27b673af4922795d9edb85e`
- Source archive:
  `https://codeload.github.com/fla-org/flash-linear-attention/tar.gz/9c8e42e762fce087c27b673af4922795d9edb85e`
- Source archive SHA256:
  `79979508114abe180affdacef1d99bad88ff0a57990144390ef505ab91862771`
- Wheel: `flash_linear_attention-0.5.2-9c8e42e-py3-none-any.whl`
- Wheel SHA256:
  `0280db310981915eb048ece99d7bedca8b5caa9be65c99835a0f912ada977d6a`
- Build command: `uv build --wheel`

The archive was downloaded and the wheel was built locally without source
patches. `setup.sh` installs this exact repository-local wheel into a
content-addressed directory below `.cache/fla-versions/`, validates it, and
atomically switches `.cache/fla-active`. It then atomically writes the full
source commit to `.cache/fla-source-sha`. The older `fe8fce9` wheel remains only
to reproduce historical runs.
