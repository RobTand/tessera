#!/usr/bin/env bash
# Published PB admits one LOCAL action per rank for the explicit bounded plan.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec "${RUNTIME_IMAGE_PY:-python3}" "$HERE/window_driver.py" "$@"
