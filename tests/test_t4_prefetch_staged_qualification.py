"""CPU control proofs only; no mocked kernel output or GPU qualification."""
import os
from types import SimpleNamespace

import pytest

from experiments.t4_code.prefetch_qualification import finish_banks


@pytest.mark.parametrize("fault", ["bind", "hash"])
def test_first_owner_failure_closes_both_real_fds_before_reader_release(tmp_path, fault):
    paths = [tmp_path / "first.so", tmp_path / "second.so"]
    for path in paths:
        path.write_bytes(b"actual file-backed control descriptor")
    fds = [os.open(path, os.O_RDONLY) for path in paths]
    events = []

    class Owner:
        def __init__(self, index):
            self.fd = fds[index]
            self.module = object()
            self.closed = False
            self.index = index

        def attest_mapped(self, module):
            if self.index == 0 and fault == "bind":
                raise ValueError("first-owner bind failure")

        def finish(self, fence, *, keep_load_fd):
            fence()
            events.append(self.index)
            self.closed = True
            if self.index == 0 and fault == "hash":
                raise ValueError("first-owner hash failure")

    def release():
        for fd in fds:
            with pytest.raises(OSError):
                os.fstat(fd)
        events.append("released")

    reader = SimpleNamespace(close=release)
    primary = RuntimeError("original consume failure")
    finish_banks(reader, [Owner(0), Owner(1)], lambda: None, primary=primary)
    assert events == [0, 1, "released"]
    assert any(f"first-owner {fault} failure" in note for note in primary.__notes__)


def test_no_primary_failure_reports_cleanup_error_after_both_fds_close(tmp_path):
    path = tmp_path / "owned.so"
    path.write_bytes(b"file-backed control")
    fds = [os.open(path, os.O_RDONLY), os.open(path, os.O_RDONLY)]
    released = []

    def owner(fd, fail):
        state = SimpleNamespace(fd=fd, module=object(), closed=False,
                                attest_mapped=lambda module: None)
        def finish(fence, *, keep_load_fd):
            fence()
            state.closed = True
            if fail:
                raise ValueError("retained hash changed")
        state.finish = finish
        return state

    with pytest.raises(ValueError, match="retained hash changed"):
        finish_banks(SimpleNamespace(close=lambda: released.append(True)),
                     [owner(fds[0], True), owner(fds[1], False)], lambda: None)
    assert released == [True]
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("mode", ["native", "tool"])
def test_post_acquire_directory_failure_releases_reader(tmp_path, monkeypatch, mode):
    from experiments.t4_code import prefetch_qualification as qualified
    import pb_staged_store

    released = []
    reader = SimpleNamespace(close=lambda: released.append(True))
    monkeypatch.setattr(qualified, "validate_readset", lambda path: None)
    monkeypatch.setattr(pb_staged_store, "StagedInputs", lambda path: reader)
    if mode == "native":
        target = tmp_path / "not-a-directory"
        target.write_bytes(b"real directory-creation failure")
        with pytest.raises(FileExistsError):
            with qualified.leased_banks("declared-control", target, gpu=False):
                pytest.fail("must not yield after setup failure")
    else:
        target = tmp_path / "out"
        target.mkdir()
        (target / "staged-sanitizer").write_bytes(b"real directory-creation failure")
        with pytest.raises(FileExistsError):
            qualified.sanitize(SimpleNamespace(manifest="declared-control", out=str(target),
                                               command="tool-preflight"))
    assert released == [True]


@pytest.mark.parametrize("fault", [None, "offset", "duplicate-path", "order"])
def test_public_manifest_validator_enforces_exact_whole_file_members(tmp_path, fault):
    import json
    from experiments.t4_code import prefetch_qualification as qualified

    entries = [{"path": str(qualified.NATIVE_RECORD), "offset": 0, "bytes": 852,
                "sha256": "b8b242cbbb2901320942e87a273cbce024e86276ad72ab66b23fe5778eb6d9d0"}]
    for arm, (module, digest) in qualified.ARMS.items():
        entries.append({"path": str(qualified.NATIVE_ROOT / f"build_apf{arm}" / (module + ".so")),
                        "offset": 0, "bytes": 1889616, "sha256": digest})
    if fault == "offset":
        entries[0]["offset"] = 1
    elif fault == "duplicate-path":
        entries.append(dict(entries[0], offset=1, bytes=1))
    elif fault == "order":
        entries.reverse()
    total = sum(row["bytes"] for row in entries)
    value = {"schema": "prismaquant.prismabuild.data_manifest.v1", "produced_by": {"test": "public API"},
             "mount_prefix": "/mnt/shared", "entries": entries, "entry_count": len(entries),
             "total_bytes": total, "annotations": {"phases": [
                 {"name": qualified.PHASE, "bytes": total, "cumulative_bytes": total}]}}
    path = tmp_path / "declared.json"
    path.write_text(json.dumps(value))
    if fault is None:
        normalized, phases = qualified.validate_readset(path)
        assert normalized["entries"] == entries
        assert phases == [{"name": qualified.PHASE, "start_bytes": 0, "end_bytes": total}]
    else:
        with pytest.raises(ValueError, match="offset-zero|member order"):
            qualified.validate_readset(path)
