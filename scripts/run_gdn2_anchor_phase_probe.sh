#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/scripts/run_gdn2_address_arm.sh" anchor_phase "${1:-formal}"
