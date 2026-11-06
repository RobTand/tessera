"""Check commit records in a synthetic install directory."""

import pytest


def test_directory_install_rejects_neighbor_commit(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from tessera import DISTRIBUTION, serving_parts

    holder = tmp_path / "site-packages"
    package = holder / "tessera"
    package.mkdir(parents=True)
    monkeypatch.setattr(serving_parts, "__file__", str(package / "serving_parts.py"))
    own = holder / "tessera_quant-0.1.dist-info"
    own.mkdir()
    (own / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {DISTRIBUTION}\nVersion: 0.1\n")
    (own / "direct_url.json").write_text(json.dumps({
        "url": "file:///source/tessera", "dir_info": {}}))
    neighbor = holder / "other_producer-0.1.dist-info"
    neighbor.mkdir()
    (neighbor / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: other-producer\nVersion: 0.1\n")
    (neighbor / "direct_url.json").write_text(json.dumps({
        "url": "https://example.invalid/other-producer.git",
        "vcs_info": {"vcs": "git", "commit_id": "b" * 40}}))

    assert serving_parts._installed_commit_id() is None, (
        "directory-installed Tessera must not borrow a neighboring distribution's commit")

    def no_git(*args, **kwargs):
        raise FileNotFoundError("no Git checkout")

    monkeypatch.setattr(serving_parts, "subprocess", SimpleNamespace(check_output=no_git))
    monkeypatch.delenv("TESSERA_GIT", raising=False)
    with pytest.raises(SystemExit, match="^cannot stamp the Tessera commit"):
        serving_parts.git_hash()


@pytest.mark.parametrize("distribution_name", [
    "tessera-quant", "Tessera_Quant", "TESSERA.QUANT", "TeSsErA-_.QuAnT",
])
def test_own_git_commit_uses_normalized_name(tmp_path, monkeypatch, distribution_name):
    import json
    from types import SimpleNamespace

    from tessera import serving_parts

    holder = tmp_path / "site-packages"
    package = holder / "tessera"
    package.mkdir(parents=True)
    monkeypatch.setattr(serving_parts, "__file__", str(package / "serving_parts.py"))
    own = holder / "tessera_quant-0.1.dist-info"
    own.mkdir()
    (own / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution_name}\nVersion: 0.1\n")
    commit = "a" * 40
    (own / "direct_url.json").write_text(json.dumps({
        "url": "https://example.invalid/tessera.git",
        "vcs_info": {"vcs": "git", "commit_id": commit}}))

    assert serving_parts._installed_commit_id() == commit

    def no_git(*args, **kwargs):
        raise FileNotFoundError("no Git checkout")

    monkeypatch.setattr(serving_parts, "subprocess", SimpleNamespace(check_output=no_git))
    monkeypatch.delenv("TESSERA_GIT", raising=False)
    assert serving_parts.git_hash() == commit


def test_install_search_stays_in_imported_holder(tmp_path, monkeypatch):
    import json

    from tessera import DISTRIBUTION, serving_parts

    holder = tmp_path / "site-packages"
    package = holder / "tessera"
    package.mkdir(parents=True)
    monkeypatch.setattr(serving_parts, "__file__", str(package / "serving_parts.py"))
    other_holder = tmp_path / "other-site-packages"
    own = other_holder / "tessera_quant-0.1.dist-info"
    own.mkdir(parents=True)
    (own / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {DISTRIBUTION}\nVersion: 0.1\n")
    (own / "direct_url.json").write_text(json.dumps({
        "url": "https://example.invalid/tessera.git",
        "vcs_info": {"vcs": "git", "commit_id": "c" * 40}}))
    monkeypatch.syspath_prepend(str(other_holder))

    assert serving_parts._installed_commit_id() is None


@pytest.mark.parametrize("source", ["git", "environment", "install"])
def test_git_hash_source_order(monkeypatch, source):
    from types import SimpleNamespace

    from tessera import serving_parts

    calls = []

    def git(*args, **kwargs):
        calls.append("git")
        if source == "git":
            return "git-commit\n"
        raise FileNotFoundError("no Git checkout")

    def installed():
        calls.append("install")
        return "install-commit"

    monkeypatch.setattr(serving_parts, "subprocess", SimpleNamespace(check_output=git))
    monkeypatch.setattr(serving_parts, "_installed_commit_id", installed)
    if source == "install":
        monkeypatch.delenv("TESSERA_GIT", raising=False)
    else:
        monkeypatch.setenv("TESSERA_GIT", "environment-commit")

    assert serving_parts.git_hash() == f"{source}-commit"
    assert calls == (["git", "install"] if source == "install" else ["git"])
