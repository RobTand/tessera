#!/usr/bin/env bash
# A campaign submits each scored run separately and waits before the next arm.
_FLUX_SERVE_CAMPAIGN_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
flux_serve_score() { python3 "$_FLUX_SERVE_CAMPAIGN_ROOT/tools/_flux_serve.py" dispatch "$@"; }
