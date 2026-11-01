"""CPU control proofs only; no mocked kernel output or GPU qualification."""
import os
from types import SimpleNamespace

import pytest

from experiments.t4_code.prefetch_qualification import finish_banks
from _whole_phase_sdk import whole_phase_sdk, manifest_value, write_manifest


def _native_entries(qualified, tools=False):
    entries = [{"path": path, "offset": 0, "bytes": size, "sha256": digest}
               for path, (size, digest) in qualified.native_members().items()]
    if tools:
        entries = [{"path": str(qualified.TOOLS_ROOT / name), "offset": 0,
                    "bytes": 7, "sha256": digest}
                   for name, digest in qualified.TOOLS_SHA.items()] + entries
    return entries


@pytest.mark.parametrize("schema,variant", [
    ("v1", "whole"), ("v1", "trailing-zero"), ("v1", "v1-advisory-bytes"),
    ("v2", "whole"), ("v2", "trailing-zero"), ("v2", "resident"),
])
@pytest.mark.parametrize("tools", [False, True])
def test_whole_phase_public_schema_controls(tmp_path, whole_phase_sdk, schema, variant, tools):
    """No artifact reads: members are declarations, including nonexistent tools."""
    from experiments.t4_code import prefetch_qualification as qualified

    entries = _native_entries(qualified, tools)
    value = manifest_value(entries, qualified.PHASE, schema, variant)
    path = write_manifest(tmp_path / "declared.json", value)
    validated, encoding = whole_phase_sdk.read_data_manifest(path)
    assert encoding == "identity"
    assert whole_phase_sdk.manifest_read_entries(validated) == entries
    normalized, phases = qualified.validate_readset(path)
    assert normalized == validated
    assert phases == [{"name": qualified.PHASE, "start_bytes": 0,
                       "end_bytes": sum(row["bytes"] for row in entries)}]


@pytest.mark.parametrize("schema", ["v1", "v2"])
@pytest.mark.parametrize("variant", ["missing", "wrong", "split", "multiple", "incomplete", "leading-zero"])
def test_whole_phase_public_phase_refusals(tmp_path, whole_phase_sdk, schema, variant):
    from experiments.t4_code import prefetch_qualification as qualified

    entries = _native_entries(qualified)
    path = write_manifest(tmp_path / "declared.json", manifest_value(entries, qualified.PHASE, schema))
    qualified.validate_readset(path)  # A refusal is meaningful only beside an accepted control.
    write_manifest(path, manifest_value(entries, qualified.PHASE, schema, variant))
    with pytest.raises(ValueError):
        qualified.validate_readset(path)


@pytest.mark.parametrize("fault", ["native-size", "native-digest", "tool-digest", "tool-order",
                                   "partial-tools", "extra-member", "v2-wrong-read-order", "gzip"])
def test_whole_phase_retains_exact_member_bindings(tmp_path, whole_phase_sdk, fault):
    import gzip
    from experiments.t4_code import prefetch_qualification as qualified

    entries = _native_entries(qualified, tools=True)
    schema = "v2" if fault == "v2-wrong-read-order" else "v1"
    path = write_manifest(tmp_path / "declared.json", manifest_value(entries, qualified.PHASE, schema))
    qualified.validate_readset(path)
    if fault == "native-size":
        entries[-1]["bytes"] += 1
    elif fault == "native-digest":
        entries[-1]["sha256"] = "0" * 64
    elif fault == "tool-digest":
        entries[0]["sha256"] = "0" * 64
    elif fault == "tool-order":
        entries[0], entries[1] = entries[1], entries[0]
    elif fault == "partial-tools":
        entries.pop(0)
    elif fault == "extra-member":
        entries.append(dict(path=str(qualified.NATIVE_ROOT / "foreign.so"), offset=0, bytes=1, sha256="0" * 64))
    value = manifest_value(entries, qualified.PHASE, schema)
    if fault == "v2-wrong-read-order":
        value["read_plan"]["phases"][0]["entry_indices"].reverse()
    write_manifest(path, value)
    if fault == "gzip":
        path.write_bytes(gzip.compress(path.read_bytes()))
    with pytest.raises(ValueError):
        qualified.validate_readset(path)


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
        pytest.importorskip("torch", reason="native FD owner control requires Torch; pure metadata tests do not")
        import torch
        if torch.cuda.is_available():
            pytest.skip("the CPU native-map proof requires GPU visibility disabled and "
                        "this session shows a visible CUDA device (tessera#939)")
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
def test_public_manifest_validator_enforces_exact_whole_file_members(tmp_path, fault, whole_phase_sdk):
    import json
    from experiments.t4_code import prefetch_qualification as qualified

    entries = [{"path": path, "offset": 0, "bytes": size, "sha256": digest}
               for path, (size, digest) in qualified.native_members().items()]
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


@pytest.mark.parametrize("arm", [0, 4])
@pytest.mark.parametrize("fault", [None, "source", "image", "selector", "module", "pending", "final-source", "final-elf", "action"])
def test_actual_finalized_production_bank_metadata_fails_closed(arm, fault):
    import hashlib
    import json
    from pathlib import Path
    from experiments.t4_code import prefetch_qualification as qualified

    fixture_root = Path(__file__).resolve().parents[1] / "experiments/results"
    payloads = []
    for name in ("record", "finalization"):
        raw = (fixture_root / f"t4_875_composed_{name}{arm}.json").read_bytes()
        assert len(raw) == qualified.BANKS[arm][name + "_bytes"]
        assert hashlib.sha256(raw).hexdigest() == qualified.BANKS[arm][name + "_sha256"]
        payloads.append(json.loads(raw))
    record, finalization = payloads
    if fault == "source":
        record["source_sha256"] = "0" * 64
    elif fault == "image":
        record["image"]["resolved_reference"] = record["image"]["pinned"]
    elif fault == "selector":
        record["selectors"]["TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH"] = "1"
    elif fault == "module":
        record["libraries"][0]["module"] = "tessera_routed_fused_e2m1_dev_apf0"
    elif fault == "pending":
        finalization["no_pending_work"] = False
    elif fault == "final-source":
        finalization["bindings"][record["source_path"]]["sha256"] = "0" * 64
    elif fault == "final-elf":
        finalization["bindings"][record["libraries"][0]["path"]]["bytes"] += 1
    elif fault == "action":
        record["action_key"] = "0" * 64
    if fault is None:
        qualified.validate_bank_record(record, finalization, arm, source_module="tessera_routed_fused_value")
    else:
        with pytest.raises(ValueError):
            qualified.validate_bank_record(record, finalization, arm, source_module="tessera_routed_fused_value")


@pytest.mark.parametrize("arm", [0, 4])
def test_native_cpu_map_checks_runner_before_mapping(tmp_path, monkeypatch, arm):
    import sys
    from experiments.t4_code import prefetch_qualification as qualified

    monkeypatch.setitem(sys.modules, "pytest", None)
    def unexpected_mapping(*args, **kwargs):
        raise AssertionError("missing runner must fail before native mapping")
    monkeypatch.setattr(qualified, "leased_banks", unexpected_mapping)
    with pytest.raises(ModuleNotFoundError, match="pytest"):
        qualified.consume(SimpleNamespace(
            command="consume", manifest=str(tmp_path / "readset.json"),
            out=str(tmp_path / "native"), arm=arm, cpu_map=True,
        ))
