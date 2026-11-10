#!/usr/bin/env python3
"""Launch the finite rootless container population inside its native Flux job."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _suite_container import main
import _suite_flux
from _suite_runtime import identity

if __name__ == "__main__":
    raise SystemExit(main(transport=_suite_flux, runtime_identity=identity))
