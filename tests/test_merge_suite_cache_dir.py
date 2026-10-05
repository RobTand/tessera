"""Cache filesystem admission happens before any suite action is submitted."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "merge_suite.py"


class SubmissionReached(Exception):
    pass


def _mount(mount_id, mountpoint, filesystem):
    escaped = str(mountpoint).replace("\\", r"\134").replace(" ", r"\040")
    return f"{mount_id} 1 0:{mount_id} / {escaped} rw - {filesystem} source rw\n"


@pytest.fixture
def admission(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("_merge_suite_cache", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "DEFAULT_RECEIPT_ROOT", tmp_path / "receipts")
    original_read = Path.read_text
    mount_table = []

    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/mountinfo":
            return "".join(mount_table)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)

    def submit(*args, **kwargs):
        raise SubmissionReached("suite submission was reached")

    monkeypatch.setattr(module, "_submit", submit)

    def run(cache, mounts):
        mount_table[:] = mounts
        monkeypatch.setattr(sys, "argv", [str(TOOL), "--arm", "gpu",
            "--gpu-tag", "fake-worker", "--checkout", str(ROOT),
            "--gpu-image", "sha256:" + "a" * 64,
            "--gpu-deps-site", str(tmp_path / "dependencies"),
            "--gpu-deps-sha256", "b" * 64, "--gpu-cache-dir", str(cache)])
        return module.main()

    return run


@pytest.mark.parametrize("filesystem", ["nfs", "nfs4", "cifs", "smb3", "fuse.sshfs"])
def test_network_cache_refuses_before_submission(tmp_path, admission, capsys, filesystem):
    with pytest.raises(SystemExit) as error:
        admission(tmp_path, [_mount(1, "/", filesystem)])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "--gpu-cache-dir" in message
    assert str(tmp_path) in message
    assert "mount point /" in message
    assert f"filesystem type {filesystem}" in message
    assert "local disk" in message
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize("filesystem", ["ext4", "xfs", "btrfs", "tmpfs"])
def test_local_cache_is_admitted(tmp_path, admission, filesystem):
    with pytest.raises(SubmissionReached):
        admission(tmp_path, [_mount(1, "/", filesystem)])


def test_nested_network_mount_wins_over_local_root(tmp_path, admission, capsys):
    mountpoint = tmp_path / "network cache"
    mountpoint.mkdir()
    cache = mountpoint / "cache"
    cache.mkdir()
    # Deliberately put the parent last: matching must follow depth, not order.
    with pytest.raises(SystemExit):
        admission(cache, [_mount(2, mountpoint, "nfs"), _mount(1, "/", "ext4")])
    assert f"mount point {mountpoint}" in capsys.readouterr().err


@pytest.mark.parametrize("filesystem,refused", [("nfs4", True), ("ext4", False)])
def test_missing_cache_uses_nearest_existing_parent(tmp_path, admission, capsys, filesystem, refused):
    cache = tmp_path / "not yet created" / "cache"
    assert not cache.exists()
    # The deeper fake mount does not exist; it must not determine admission.
    mounts = [_mount(1, "/", "ext4"), _mount(2, tmp_path, filesystem),
              _mount(3, cache.parent, "ext4" if refused else "nfs")]
    if refused:
        with pytest.raises(SystemExit):
            admission(cache, mounts)
        message = capsys.readouterr().err
        assert str(cache) in message
        assert f"mount point {tmp_path}" in message
        assert f"filesystem type {filesystem}" in message
    else:
        with pytest.raises(SubmissionReached):
            admission(cache, mounts)
    assert not cache.exists()


def test_network_mount_prefix_does_not_match_sibling(tmp_path, admission):
    mountpoint = tmp_path / "network"
    mountpoint.mkdir()
    cache = tmp_path / "network-local"
    cache.mkdir()
    with pytest.raises(SubmissionReached):
        admission(cache, [_mount(1, "/", "ext4"), _mount(2, mountpoint, "nfs")])


def test_cache_symlink_is_judged_by_its_target(tmp_path, admission, capsys):
    mountpoint = tmp_path / "network"
    mountpoint.mkdir()
    alias = tmp_path / "local-alias"
    alias.symlink_to(mountpoint, target_is_directory=True)
    with pytest.raises(SystemExit):
        admission(alias / "cache", [_mount(1, "/", "ext4"), _mount(2, mountpoint, "nfs")])
    assert f"mount point {mountpoint}" in capsys.readouterr().err


@pytest.mark.parametrize("mounts", [[], ["not a mount table\n"]])
def test_unknown_mount_provenance_refuses(tmp_path, admission, capsys, mounts):
    with pytest.raises(SystemExit):
        admission(tmp_path, mounts)
    message = capsys.readouterr().err
    assert "--gpu-cache-dir" in message
    assert str(tmp_path) in message
    assert "mount" in message
