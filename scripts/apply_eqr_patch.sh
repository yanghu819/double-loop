#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EQR_DIR="${EQR_DIR:-$REPO_ROOT/repos/eqr}"
EQR_URL="${EQR_URL:-https://github.com/locuslab/eqr.git}"
EQR_BASE_SHA="${EQR_BASE_SHA:-aba94e9cde0f273ce644db5261cd6915ba6561f0}"
PATCH_PATH="${PATCH_PATH:-$REPO_ROOT/docs/eqr_patches/0001-Add-FutureSeed-and-feature-diff-noise-to-EqR.patch}"

mkdir -p "$(dirname "$EQR_DIR")"

if [[ ! -d "$EQR_DIR/.git" ]]; then
  git clone "$EQR_URL" "$EQR_DIR"
fi

if [[ -n "$(git -C "$EQR_DIR" status --short --untracked-files=no)" ]]; then
  current_sha="$(git -C "$EQR_DIR" rev-parse HEAD)"
  if [[ "$current_sha" == "$EQR_BASE_SHA" ]] && git -C "$EQR_DIR" apply --reverse --check "$PATCH_PATH" >/dev/null 2>&1; then
    git -C "$EQR_DIR" rev-parse HEAD > "$EQR_DIR/.double-loop-base-sha"
    printf 'EqR FutureSeed/feature-noise patch already applied at %s\n' "$EQR_DIR"
    printf 'EqR ready: base=%s dir=%s\n' "$EQR_BASE_SHA" "$EQR_DIR"
    exit 0
  fi
  printf 'EqR repo has tracked local changes; refusing to overwrite: %s\n' "$EQR_DIR" >&2
  git -C "$EQR_DIR" status --short >&2
  exit 1
fi

git -C "$EQR_DIR" fetch --quiet origin "$EQR_BASE_SHA" || true
git -C "$EQR_DIR" checkout --detach "$EQR_BASE_SHA"

if git -C "$EQR_DIR" apply --reverse --check "$PATCH_PATH" >/dev/null 2>&1; then
  printf 'EqR FutureSeed/feature-noise patch already applied at %s\n' "$EQR_DIR"
elif git -C "$EQR_DIR" apply --check "$PATCH_PATH"; then
  git -C "$EQR_DIR" apply "$PATCH_PATH"
  printf 'Applied EqR FutureSeed/feature-noise patch at %s\n' "$EQR_DIR"
else
  printf 'EqR patch does not apply cleanly: %s\n' "$PATCH_PATH" >&2
  exit 1
fi

git -C "$EQR_DIR" rev-parse HEAD > "$EQR_DIR/.double-loop-base-sha"
printf 'EqR ready: base=%s dir=%s\n' "$EQR_BASE_SHA" "$EQR_DIR"
