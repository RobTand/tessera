"""Only the Tessera distribution can supply its installed commit."""


def test_unrelated_vcs_metadata_does_not_stamp_tessera(tmp_path):
    """Refuse a directory install even when another package records a commit."""
    __tracebackhide__ = True
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys

    from tessera import DISTRIBUTION

    root = Path(__file__).resolve().parents[1]
    site = tmp_path / "site-packages"
    site.mkdir()
    shutil.copytree(
        root / "src" / "tessera", site / "tessera",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    owner = site / f"{DISTRIBUTION.replace('-', '_')}-0.1.dist-info"
    owner.mkdir()
    (owner / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {DISTRIBUTION}\nVersion: 0.1\n"
    )
    (owner / "top_level.txt").write_text("tessera\n")
    (owner / "direct_url.json").write_text(json.dumps({
        "url": "file:///local/tessera", "dir_info": {},
    }))
    other = site / "unrelated_package-0.1.dist-info"
    other.mkdir()
    (other / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: unrelated-package\nVersion: 0.1\n"
    )
    (other / "top_level.txt").write_text("unrelated_package\n")
    other_commit = "a" * 40
    (other / "direct_url.json").write_text(json.dumps({
        "url": "https://example.invalid/unrelated.git",
        "vcs_info": {"vcs": "git", "commit_id": other_commit},
    }))
    program = (
        "import json, tessera\n"
        "from tessera.serving_parts import _installed_commit_id, git_hash\n"
        "result = {'module': tessera.__file__, 'installed': _installed_commit_id()}\n"
        "try:\n"
        "    result['git_hash'] = git_hash()\n"
        "except SystemExit as exc:\n"
        "    result['refusal'] = str(exc)\n"
        "print(json.dumps(result))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(site)
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    env.pop("TESSERA_GIT", None)
    proc = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert Path(result["module"]).resolve() == site / "tessera" / "__init__.py"
    assert result["installed"] is None, (
        "_installed_commit_id() returned the commit of another distribution"
    )
    assert "git_hash" not in result, "git_hash() accepted an unrelated commit"
    assert "cannot stamp the Tessera commit" in result.get("refusal", "")
