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
        assert "--paginate" in argv
        assert "repos/example/repo/issues?state=all&per_page=100" in argv
        return SimpleNamespace(stdout=json.dumps(pages))

    monkeypatch.setattr(module.subprocess, "run", run)
    found = module.fetch("example/repo")
    assert set(found) == {str(row["number"]) for page in pages for row in page}
    assert found["12"] == {"title": "merged fix", "state": "CLOSED"}
    assert found["4"]["state"] == "OPEN"


def _module():
    spec = importlib.util.spec_from_file_location(
        "refresh_issues", Path(__file__).resolve().parents[1] / "tools/refresh_issues.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(monkeypatch, tmp_path, snapshot, fresh):
    """Run ``--check`` against a committed snapshot and a fetch that answers."""

    module = _module()
    path = tmp_path / "issues-snapshot.json"
    path.write_text(json.dumps({
        "generated": "2026-09-05T00:00:00Z",
        "default_repo": "example/repo",
        "repos": {"example/repo": snapshot},
    }))
    monkeypatch.setattr(module, "SNAPSHOT", path)
    monkeypatch.setattr(module, "REPOS", ("example/repo",))
    monkeypatch.setattr(module, "fetch", lambda repo: fresh)
    monkeypatch.setattr(module.sys, "argv", ["refresh_issues.py", "--check"])
    return module.main()


def test_check_refuses_an_id_the_source_no_longer_has(monkeypatch, tmp_path, capsys):
    """#220: a deleted or transferred issue stayed 'current' forever.

    ``--check`` compared fresh-minus-known and the states of the intersection,
    and never looked at known-minus-fresh. The offline reference gate builds
    its allowed IDs straight out of that snapshot, so every documentation
    reference to an issue that has left the repository kept resolving.
    """

    status = _check(monkeypatch, tmp_path,
                    snapshot={"7": {"title": "still here", "state": "OPEN"},
                              "8": {"title": "transferred away", "state": "OPEN"}},
                    fresh={"7": {"title": "still here", "state": "OPEN"}})

    assert status == 1
    assert "8" in capsys.readouterr().err


def test_check_accepts_a_snapshot_that_matches_its_source(monkeypatch, tmp_path):
    same = {"7": {"title": "still here", "state": "OPEN"}}
    assert _check(monkeypatch, tmp_path, snapshot=same, fresh=dict(same)) == 0


def test_check_still_refuses_a_new_id_and_a_changed_state(monkeypatch, tmp_path):
    known = {"7": {"title": "still here", "state": "OPEN"}}
    assert _check(monkeypatch, tmp_path, snapshot=known, fresh={
        **known, "9": {"title": "filed since", "state": "OPEN"}}) == 1
    assert _check(monkeypatch, tmp_path, snapshot=known, fresh={
        "7": {"title": "still here", "state": "CLOSED"}}) == 1


def test_fetch_reads_the_pages_gh_prints_without_slurp(monkeypatch):
    """`--slurp` is gh 2.53 and later; this box has 2.45.

    Without it `gh api --paginate` concatenates one array per page, which is
    not a JSON document -- and requiring the flag made the refresher exit
    `unknown flag: --slurp`, so the snapshot the offline reference gate reads
    could not be regenerated from this box at all.
    """

    module = _module()
    concatenated = (
        '[{"number": 4, "title": "open finding", "state": "open"}]\n'
        '[{"number": 9, "title": "older finding", "state": "closed"}]\n')

    monkeypatch.setattr(module.subprocess, "run",
                        lambda argv, **kw: SimpleNamespace(stdout=concatenated))
    found = module.fetch("example/repo")
    assert set(found) == {"4", "9"}
    assert found["9"] == {"title": "older finding", "state": "CLOSED"}
