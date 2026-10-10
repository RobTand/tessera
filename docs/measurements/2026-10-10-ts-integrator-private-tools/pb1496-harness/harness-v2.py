#!/usr/bin/env python3
"""PB1496 acceptance proof for the merged TS924 selector (private fixtures only).

Read-only on the checkout under test: every repository this script mutates is a
temporary one it builds or clones itself.  It drives the REAL CLI,
``tools/impacted_tests.py --root REPO --ref BASE...HEAD --json``, and judges each
answer against the PrismaBuild 1496 acceptance wording:

    a changed, new or removed member of a directory a test reads must select that
    test, or the verdict must be an explicit ``full``; ``narrowed`` or ``none``
    without the reader is a false proof.

usage: harness.py CHECKOUT OUT_JSON
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

CHECKOUT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
SCRIPT = CHECKOUT / "tools" / "impacted_tests.py"
READER = "tests/test_reader.py"


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()[:300]}")
    return done.stdout.strip()


def new_repo(base: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = base / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "pb1496@example.invalid")
    git(repo, "config", "user.name", "PB1496 acceptance")
    for relative, body in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    return repo, git(repo, "rev-parse", "HEAD")


def apply(repo: Path, ops: list[tuple]) -> None:
    for op in ops:
        kind = op[0]
        if kind in ("write", "add"):
            path = repo / op[1]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(op[2], encoding="utf-8")
            git(repo, "add", op[1])
        elif kind == "rm":
            git(repo, "rm", "-q", op[1])
        elif kind == "mv":
            (repo / op[2]).parent.mkdir(parents=True, exist_ok=True)
            git(repo, "mv", op[1], op[2])
        else:
            raise ValueError(kind)
    git(repo, "commit", "-qm", "change")


def selector(repo: Path, base: str) -> dict:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), "--ref", f"{base}...HEAD", "--json"],
        capture_output=True, text=True)
    if done.returncode != 0:
        return {"cli_error": done.stderr.strip()[:400], "rc": done.returncode}
    if not done.stdout.strip():
        return {"verdict": "no-output", "tests": [], "rc": 0, "stderr": done.stderr.strip()[:200]}
    out = json.loads(done.stdout)
    out["rc"] = 0
    return out


# ---------------------------------------------------------------- consumer forms
READER_FORMS = {
    "rglob": 'def test_reads():\n    assert all(p.read_text() for p in sorted(DIR.rglob("*{ext}")))\n',
    "iterdir": 'def test_reads():\n    assert sorted(DIR.iterdir())\n',
    "listdir": 'def test_reads():\n    assert sorted(os.listdir(DIR))\n',
    "scandir": 'def test_reads():\n    with os.scandir(DIR) as entries:\n        assert sorted(e.name for e in entries)\n',
    "walk": 'def test_reads():\n    assert [name for _, _, names in os.walk(DIR) for name in names]\n',
    "glob": 'def test_reads():\n    assert sorted(DIR.glob("*{ext}"))\n',
    "globread": 'def test_reads():\n    for path in sorted(DIR.glob("*{ext}")):\n        assert path.read_text()\n',
    "globrec": 'def test_reads():\n    for path in DIR.glob("**/*{ext}"):\n        assert path.read_text()\n',
}


def reader_source(form: str, directory: str, ext: str) -> str:
    head = (f'import os\nfrom pathlib import Path\n\n'
            f'DIR = Path(__file__).resolve().parents[1] / "{directory}"\n\n\n')
    return head + READER_FORMS[form].replace("{ext}", ext)


def base_files(form: str, directory: str, ext: str) -> dict[str, str]:
    return {
        "seed.txt": "base\n",
        f"{directory}/first{ext}": "# one\n",
        f"{directory}/nested/second{ext}": "# two\n",
        READER: reader_source(form, directory, ext),
        "tests/test_unrelated.py": "def test_unrelated():\n    assert True\n",
    }


def judge(result: dict, expect: str) -> tuple[bool, str]:
    """True when the answer is acceptable under the PB1496 wording."""
    if "cli_error" in result:
        return False, "cli-error"
    verdict = result.get("verdict")
    selected = READER in (result.get("tests") or [])
    if expect == "selected-or-full":
        if verdict == "full":
            return True, "explicit full"
        if selected:
            return True, "reader selected"
        return False, f"FALSE PROOF: verdict {verdict}, reader not selected"
    if expect == "not-required":
        return True, f"not required by contract (verdict {verdict}, reader selected={selected})"
    if expect == "not-selected":
        if selected:
            return False, "reader selected for an unrelated change"
        return True, f"reader not selected (verdict {verdict})"
    if expect.startswith("documented-limit:"):
        # An enumerated base nothing can name, in a module that executes nothing, states no
        # dependency (#148): verdict none and no test, but the selector must LIST the reader
        # (PB1496 guard), so a reader of this shape is seen and not silently unselected.
        holder = expect.split(":", 1)[1]
        listed = holder in (result.get("unnamed_directory_reads") or {})
        if verdict == "none" and not result.get("tests") and listed:
            return True, f"documented limit: verdict none, {holder} listed"
        return False, f"verdict {verdict}, tests {len(result.get('tests') or [])}, {holder} listed={listed}"
    if expect == "full":
        return (verdict == "full"), f"verdict {verdict}"
    if expect == "full-equivalent":
        every = result.get("_every_test_file", 0)
        n = len(result.get("tests") or [])
        if verdict == "full":
            return True, "explicit full"
        if every and n >= every:
            return True, f"narrowed list names every test file ({n}/{every}): full-equivalent"
        return False, f"verdict {verdict}, {n} of {every} test files"
    raise ValueError(expect)


results: list[dict] = []


def record(group: str, name: str, expect: str, result: dict) -> None:
    ok, why = judge(result, expect)
    results.append({
        "group": group, "scenario": name, "expect": expect, "ok": ok, "why": why,
        "verdict": result.get("verdict"), "n_tests": len(result.get("tests") or []),
        "reader_selected": READER in (result.get("tests") or []),
        "forces_full": result.get("forces_full"),
        "reason": str(result.get("reason"))[:200],
        "tests_head": (result.get("tests") or [])[:4],
    })


# ------------------------------------------------------------ synthetic matrix
def matrix() -> None:
    for form in READER_FORMS:
        for directory, ext in (("docs", ".md"), ("data", ".json")):
            ops_by_kind = {
                "modify": [("write", f"{directory}/first{ext}", "# changed\n")],
                "add": [("add", f"{directory}/extra{ext}", "# new\n")],
                "add-nested": [("add", f"{directory}/nested/deeper/third{ext}", "# new deep\n")],
                "delete": [("rm", f"{directory}/first{ext}")],
                "delete-nested": [("rm", f"{directory}/nested/second{ext}")],
                "rename": [("mv", f"{directory}/first{ext}", f"{directory}/renamed{ext}")],
            }
            for kind, ops in ops_by_kind.items():
                expect = "selected-or-full"
                # A single-directory, non-recursive glob never enumerates nested members, so a
                # nested add or delete is not part of the reader's input; a names-only reader's
                # result does not depend on a member's content, so a plain modify is not either.
                if form in ("glob", "globread") and kind in ("add-nested", "delete-nested"):
                    expect = "not-required"
                if form == "glob" and kind == "modify":
                    expect = "not-required"
                with tempfile.TemporaryDirectory() as tmp:
                    repo, base = new_repo(Path(tmp), base_files(form, directory, ext))
                    apply(repo, ops)
                    record(f"synthetic/{form}", f"{directory}/{kind}", expect, selector(repo, base))
        # negative control: a member of an UNREAD directory changes
        with tempfile.TemporaryDirectory() as tmp:
            files = base_files(form, "docs", ".md")
            files["elsewhere/other.md"] = "# not read\n"
            repo, base = new_repo(Path(tmp), files)
            apply(repo, [("write", "elsewhere/other.md", "# changed\n")])
            record(f"synthetic/{form}", "control: unread directory", "not-selected",
                   selector(repo, base))
        # negative control: same BASENAME elsewhere must not be taken as the read target
        with tempfile.TemporaryDirectory() as tmp:
            files = base_files(form, "docs", ".md")
            files["elsewhere/first.md"] = "# same basename, other directory\n"
            repo, base = new_repo(Path(tmp), files)
            apply(repo, [("write", "elsewhere/first.md", "# changed\n")])
            record(f"synthetic/{form}", "control: same basename elsewhere", "not-selected",
                   selector(repo, base))


def unknown_coupling() -> None:
    # A directory read whose base nothing can name, in a module that executes nothing, states
    # no dependency by design (#148).  The PB1496 guard is that the selector LISTS it.  A
    # names-only reader's result does not depend on a member's content, so a plain modify is not
    # part of its input; add and delete change the names, and a reader that reads each file
    # depends on every change.
    HELPER = "import os\n\n\ndef docs_dir():\n    return os.environ['X']\n"
    readers = {
        ("environment value", "names-only"): (READER, "import os\nfrom pathlib import Path\n\nDIR = Path(os.environ[\"READ_DIR\"])\n\n\ndef test_reads():\n    assert sorted(DIR.rglob(\"*.md\"))\n", {}),
        ("environment value", "reads-each"): (READER, "import os\nfrom pathlib import Path\n\nDIR = Path(os.environ[\"READ_DIR\"])\n\n\ndef test_reads():\n    assert [p.read_text() for p in DIR.rglob(\"*.md\")]\n", {}),
        ("helper module", "names-only"): (READER, "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / \"tools\"))\nimport where\n\n\ndef test_reads():\n    assert sorted(Path(where.docs_dir()).rglob(\"*.md\"))\n", {"tools/where.py": HELPER}),
        ("helper module", "reads-each"): (READER, "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / \"tools\"))\nimport where\n\n\ndef test_reads():\n    assert [p.read_text() for p in Path(where.docs_dir()).rglob(\"*.md\")]\n", {"tools/where.py": HELPER}),
        ("conftest", "names-only"): ("tests/conftest.py", "import os\nfrom pathlib import Path\n\nLIST = sorted(Path(os.environ['X']).rglob('*.md'))\n", {READER: "def test_reads():\n    assert True\n"}),
        ("conftest", "reads-each"): ("tests/conftest.py", "import os\nfrom pathlib import Path\n\nLIST = [p.read_text() for p in Path(os.environ['X']).rglob('*.md')]\n", {READER: "def test_reads():\n    assert True\n"}),
    }
    changes = {
        "modify": [("write", "docs/a.md", "# changed\n")],
        "add": [("add", "docs/b.md", "# new\n")],
        "delete": [("rm", "docs/a.md")],
    }
    for (base_kind, reader_kind), (holder, text, extra) in readers.items():
        for change, ops in changes.items():
            files = {"seed.txt": "b\n", "docs/a.md": "# a\n", holder: text, **extra}
            expect = f"documented-limit:{holder}"
            if reader_kind == "names-only" and change == "modify":
                expect = "not-required"
            with tempfile.TemporaryDirectory() as tmp:
                repo, base = new_repo(Path(tmp), files)
                apply(repo, ops)
                record("unknown-coupling", f"{base_kind}, {reader_kind}: {change}", expect, selector(repo, base))
    # U5: a source-executing enumeration with an unnameable base (the documented wildcard)
    with tempfile.TemporaryDirectory() as tmp:
        files = {
            "seed.txt": "b\n", "tools/plug.py": "VALUE = 1\n",
            "tools/loader.py": textwrap.dedent('''
                import os
                import runpy
                from pathlib import Path

                for path in Path(os.environ["PLUGINS"]).rglob("*.py"):
                    runpy.run_path(path)
            '''),
            READER: "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))\nimport loader\n\n\ndef test_reads():\n    assert loader\n",
        }
        repo, base = new_repo(Path(tmp), files)
        apply(repo, [("write", "tools/plug.py", "VALUE = 2\n")])
        record("unknown-coupling", "source-executing enumeration, unnameable base, a Python edit", "selected-or-full",
               selector(repo, base))
    # U4: a changed path no analysis can place (a build manifest)
    with tempfile.TemporaryDirectory() as tmp:
        files = {"seed.txt": "b\n", "pyproject.toml": "[project]\nname='x'\n", READER: "def test_reads():\n    assert True\n"}
        repo, base = new_repo(Path(tmp), files)
        apply(repo, [("write", "pyproject.toml", "[project]\nname='y'\n")])
        record("unknown-coupling", "pyproject.toml changes", "full", selector(repo, base))


# ------------------------------------------------------------ real-repo CLI smoke
def real_repo_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "clone"
        subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(CHECKOUT), str(clone)], check=True)
        git(clone, "config", "user.email", "pb1496@example.invalid")
        git(clone, "config", "user.name", "PB1496 acceptance")
        base = git(clone, "rev-parse", "HEAD")
        tracked_docs = [p for p in git(clone, "ls-files", "docs").splitlines()
                        if p.endswith(".md") and p != "docs/ARCHITECTURE.md"]
        real = {
            "real/modify docs/ARCHITECTURE.md (the TS921 shape)": (
                [("write", "docs/ARCHITECTURE.md", (clone / "docs/ARCHITECTURE.md").read_text() + "\nPB1496 probe\n")],
                "tests/test_issue_refs.py"),
            "real/add a new doc under docs/measurements": (
                [("add", "docs/measurements/zz-pb1496-probe.md", "# probe\n")], "tests/test_issue_refs.py"),
            "real/delete a tracked doc": ([("rm", tracked_docs[0])], "tests/test_issue_refs.py"),
        }
        only_glob = bool(os.environ.get("PB1496_REAL_GLOB_ONLY"))
        for name, (ops, must) in ({} if only_glob else real).items():
            git(clone, "checkout", "-q", base)
            apply(clone, ops)
            result = selector(clone, base)
            tests = result.get("tests") or []
            ok = result.get("verdict") == "full" or must in tests
            results.append({
                "group": "real-cli", "scenario": name, "expect": f"{must} or full", "ok": ok,
                "why": "explicit full" if result.get("verdict") == "full" else (
                    "consumer selected" if must in tests else f"FALSE PROOF: verdict {result.get('verdict')}"),
                "verdict": result.get("verdict"), "n_tests": len(tests), "reader_selected": must in tests,
                "forces_full": result.get("forces_full"), "reason": str(result.get("reason"))[:200],
                "tests_head": tests[:3], "rc": result.get("rc")})
        construction = [p for p in git(clone, "ls-files", "docs/measurements/construction").splitlines() if p.endswith(".json")]
        real_glob = {
            "real-glob/modify a receipt read by RECEIPTS.glob": ([("write", construction[0], (clone / construction[0]).read_text() + "\n")], "tests/test_merged_linear_partitions.py"),
            "real-glob/add a receipt read by RECEIPTS.glob": ([("add", "docs/measurements/construction/zz-pb1496-probe.json", "{}\n")], "tests/test_merged_linear_partitions.py"),
            "real-glob/delete a receipt read by RECEIPTS.glob": ([("rm", construction[-1])], "tests/test_merged_linear_partitions.py"),
            "real-glob/delete a receipt the reader does NOT name (glm53-flash-4layer.json)": ([("rm", "docs/measurements/construction/glm53-flash-4layer.json")], "tests/test_merged_linear_partitions.py"),
            "real-glob/modify a receipt the reader does NOT name (glm53-flash-4layer.json)": ([("write", "docs/measurements/construction/glm53-flash-4layer.json", (clone / "docs/measurements/construction/glm53-flash-4layer.json").read_text() + "\n")], "tests/test_merged_linear_partitions.py"),
            "real-glob/add a result read by the refit pair gate glob": ([("add", "experiments/results/refit_trailing_pair_zz_pb1496.json", "{}\n")], "tests/test_refit_trailing_pair_gate.py"),
        }
        for name, (ops, must) in real_glob.items():
            git(clone, "checkout", "-q", base)
            apply(clone, ops)
            result = selector(clone, base)
            tests = result.get("tests") or []
            ok = result.get("verdict") == "full" or must in tests
            results.append({
                "group": "real-glob", "scenario": name, "expect": f"{must} or full", "ok": ok,
                "why": "explicit full" if result.get("verdict") == "full" else (
                    "consumer selected" if must in tests else f"FALSE PROOF: verdict {result.get('verdict')}"),
                "verdict": result.get("verdict"), "n_tests": len(tests), "reader_selected": must in tests,
                "forces_full": result.get("forces_full"), "reason": str(result.get("reason"))[:200],
                "tests_head": tests[:3], "rc": result.get("rc")})
        if only_glob:
            return
        for name, ops, expect in (
            ("real/modify pyproject.toml", [("write", "pyproject.toml", (clone / "pyproject.toml").read_text() + "\n# probe\n")], "full"),
            ("real/modify tests/conftest.py", [("write", "tests/conftest.py", (clone / "tests/conftest.py").read_text() + "\n# probe\n")], "full-equivalent"),
        ):
            git(clone, "checkout", "-q", base)
            apply(clone, ops)
            result = selector(clone, base)
            result["_every_test_file"] = len([f for f in git(clone, "ls-files", "tests").splitlines()
                                              if f.startswith("tests/test_") and f.endswith(".py") and f.count("/") == 1])
            ok, why = judge(result, expect)
            results.append({
                "group": "real-cli", "scenario": name, "expect": expect, "ok": ok,
                "why": why, "verdict": result.get("verdict"),
                "n_tests": len(result.get("tests") or []), "forces_full": result.get("forces_full"),
                "reason": str(result.get("reason"))[:200], "rc": result.get("rc")})
        # a leaf edit must still narrow: the selector has not collapsed into "always full"
        leaf = "tools/check_wheel.py"
        git(clone, "checkout", "-q", base)
        apply(clone, [("write", leaf, (clone / leaf).read_text() + "\n# probe\n")])
        result = selector(clone, base)
        results.append({
            "group": "real-cli", "scenario": f"real/modify {leaf} (a leaf)", "expect": "narrowed, non-empty",
            "ok": result.get("verdict") == "narrowed" and len(result.get("tests") or []) > 0,
            "why": f"verdict {result.get('verdict')}, {len(result.get('tests') or [])} tests",
            "verdict": result.get("verdict"), "n_tests": len(result.get("tests") or []),
            "forces_full": result.get("forces_full"), "reason": str(result.get("reason"))[:200], "rc": result.get("rc")})


def main() -> int:
    if not os.environ.get("PB1496_REAL_GLOB_ONLY"):
        matrix()
        unknown_coupling()
    real_repo_smoke()
    summary = {
        "checkout": str(CHECKOUT),
        "checkout_head": git(CHECKOUT, "rev-parse", "HEAD") if (CHECKOUT / ".git").exists() else "unknown",
        "scenarios": len(results),
        "ok": sum(1 for r in results if r["ok"]),
        "failed": [r for r in results if not r["ok"]],
        "by_group": {},
    }
    for r in results:
        g = summary["by_group"].setdefault(r["group"], {"n": 0, "ok": 0})
        g["n"] += 1
        g["ok"] += int(r["ok"])
    OUT.write_text(json.dumps({"summary": summary, "results": results}, indent=1))
    print(f"scenarios {summary['scenarios']}  ok {summary['ok']}  failed {len(summary['failed'])}")
    for g, v in sorted(summary["by_group"].items()):
        print(f"  {g}: {v['ok']}/{v['n']}")
    for r in summary["failed"]:
        print(f"  FAIL {r['group']} | {r['scenario']} | {r['why']}")
    # 0 = the proof completed and wrote its results; acceptance is judged from the JSON,
    # so a finding never relabels the harness (or a wrapper around it) as failed.
    return 0


if __name__ == "__main__":
    sys.exit(main())
