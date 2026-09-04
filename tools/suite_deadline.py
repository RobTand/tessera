#!/usr/bin/env python3
"""Run the source-sealed suite deadline helper from a PB checkout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera.suite_deadline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
