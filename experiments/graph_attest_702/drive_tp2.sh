#!/usr/bin/env bash
# Published PB admits one LOCAL action per rank, holding all three arms in one finite window.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec "${RUNTIME_IMAGE_PY:-python3}" "$HERE/window_driver.py" "$@"
