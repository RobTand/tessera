#!/usr/bin/env python3
"""Launch the finite container population inside an admitted PB action."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera._dev.suite_container import main

if __name__ == "__main__":
    raise SystemExit(main())
