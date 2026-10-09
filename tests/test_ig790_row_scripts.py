"""Measurement launchers must retain a failed child status."""
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("script", ["tools-ig790-pqbridge-row.sh", "tools-ig790-verify-row.sh"])
def test_row_propagates_child_failure(script, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / script), "/bin/false", str(tmp_path / "row")],
        cwd=root, capture_output=True, text=True, timeout=45,
    )
    assert "ROW-END rc=1" in result.stdout
    assert result.returncode == 1, result.stdout
