"""A manual gate is a dependency, but not a pytest target or coverage claim."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/impacted_tests.py"
SPEC = importlib.util.spec_from_file_location("impacted_manual_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
impacted = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(impacted)

MANUAL = "tests/test_native_a4_serving.py"
REASON = "standalone manual CUDA gate (run_gate under __main__); no pytest items; not executed by this selection"


def _tree(root: Path) -> None:
    files = {
        "library.py": "VALUE = 1\n",
        "fixture.json": "{}\n",
        "conftest.py": "",
        "tests/conftest.py": "",
        MANUAL: (
            "import library\n# fixture.json\n"
            "def run_gate(): return library.VALUE\n"
            "if __name__ == '__main__': run_gate()\n"
        ),
        "tests/test_keep.py": (
            "from test_native_a4_serving import run_gate\n# fixture.json\n"
            "def test_value(): assert run_gate() == 1\n"
        ),
    }
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


@pytest.mark.parametrize("changed", [MANUAL, "library.py", "fixture.json", "tests/conftest.py", "conftest.py"])
def test_manual_gate_is_excluded_after_every_selection_path(tmp_path, changed):
    _tree(tmp_path)
    result = impacted.select(tmp_path, [changed])
    assert result["tests"] == ["tests/test_keep.py"]
    assert result["excluded_tests"] == [{"path": MANUAL, "reason": REASON}]
    assert "manual gate" in result["reason"]
    assert result["verdict"] == ("full" if changed == "conftest.py" else "narrowed")
    if changed == "conftest.py":
        assert result["forces_full"] == ["conftest.py"]


def test_real_manual_gate_remains_a_separate_entry_point():
    result = impacted.select(ROOT, [MANUAL])
    assert MANUAL not in result["tests"]
    assert result["excluded_tests"] == [{"path": MANUAL, "reason": REASON}]
    # This explicit decision does not generalize to arbitrary files with no
    # local test defs (pytest can collect imported/generated/inherited tests).
    body = ast.parse((ROOT / MANUAL).read_text())
    definitions = [n for n in ast.walk(body) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert not any(n.name.startswith(("test_", "Test")) for n in definitions)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "run_gate" for n in definitions)
    assert 'if __name__ == "__main__":' in (ROOT / MANUAL).read_text()


def test_text_receipt_names_excluded_gate_and_reason(tmp_path, monkeypatch, capsys):
    _tree(tmp_path)
    monkeypatch.setattr(impacted, "changed_files", lambda ref, root: ([MANUAL], ref))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--root", str(tmp_path)])
    assert impacted.main() == 0
    output = capsys.readouterr().out
    assert "excluded pytest targets (1):" in output
    assert f"{MANUAL}: {REASON}" in output
    assert "selected tests (1):" in output
