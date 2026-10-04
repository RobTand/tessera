"""A population already names its producer; unrelated history is irrelevant."""
import importlib.util
from pathlib import Path


def test_resume_does_not_scan_unrelated_finished_actions(tmp_path, monkeypatch):
    tests = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("_merge_test_helpers", tests / "test_merge_suite.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    original = Path.glob
    def bounded(path, pattern):
        if path.name in ("done", "failed") and pattern == "*.json":
            raise AssertionError("resume scanned unrelated completed actions")
        return original(path, pattern)
    monkeypatch.setattr(Path, "glob", bounded)
    module, record = helpers._resumed_with(tmp_path)
    assert record["exit_status_observed"] is True
    assert record["returncode"] == 0
