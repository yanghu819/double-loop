#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BASELINE_CONFIG="$REPO_ROOT/configs/sudoku/gdn3_shared_namespace_scale.env"

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${GDN2_SCALE_PROBE:-0}" == "1" ]]; then
  export RUN_NAME="gdn3-shared-namespace-d256l12-fit-${TIMESTAMP}-${GIT_SHA:0:7}"
else
  export RUN_NAME="gdn3-shared-namespace-d256l12-s12000-${TIMESTAMP}-${GIT_SHA:0:7}"
fi

exec "$REPO_ROOT/scripts/run_canonical_gdn2_scale.sh"
