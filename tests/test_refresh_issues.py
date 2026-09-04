"""The offline reference snapshot covers GitHub's shared issue/PR namespace."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def test_fetch_keeps_issues_and_pull_requests_across_pages(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "refresh_issues", Path(__file__).resolve().parents[1] / "tools/refresh_issues.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pages = [[{"number": 4, "title": "open finding", "state": "open"},
              {"number": 12, "title": "merged fix", "state": "closed",
               "pull_request": {"url": "https://api.github.com/repos/example/repo/pulls/12"}}],
             [{"number": 9, "title": "older finding", "state": "closed"}]]

    def run(argv, **kwargs):
        if argv[1:3] == ["issue", "list"]:
            # The original refresh cannot observe PRs at all.
            return SimpleNamespace(stdout=json.dumps([
                {"number": 4, "title": "open finding", "state": "OPEN"},
                {"number": 9, "title": "older finding", "state": "CLOSED"},
            ]))
        assert argv[1] == "api"
        assert "--paginate" in argv and "--slurp" in argv
        assert "repos/example/repo/issues?state=all&per_page=100" in argv
        return SimpleNamespace(stdout=json.dumps(pages))

    monkeypatch.setattr(module.subprocess, "run", run)
    found = module.fetch("example/repo")
    assert set(found) == {str(row["number"]) for page in pages for row in page}
    assert found["12"] == {"title": "merged fix", "state": "CLOSED"}
    assert found["4"]["state"] == "OPEN"
