"""Check ordered dense records, stable row ordinals and grouped capture starts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from experiments.numerics_rows import phase_artifact as streams, row
from experiments.numerics_rows.common import Invalid
from experiments.numerics_rows.launch_state import json_digest


def stream_feature(tmp_path):
    artifact = streams.Artifact(tmp_path / "prepare", "prepare", {})
    assert artifact.document["schema"] == "tessera.dense_row_phase.v2", "feature absent: ordered case streams"
    return artifact


def test_stream_releases_each_tensor_before_the_next_case(tmp_path):
    artifact = stream_feature(tmp_path)
    scope = streams.unit_scope(row.new_unit({"m_config": {"M": 1}}), "wire")
    artifact.document["scopes"] = [scope]
    for index in range(1000):
        value = torch.tensor([index])
        reference = weakref.ref(value)
        artifact.add(scope, {"M": 1}, "dense", {"input": value}, category="sweep")
        del value
        assert reference() is None, "the writer retained a case tensor"
    artifact.finish()
    assert "cases" not in artifact.document and "files" not in artifact.document
    loaded, _ = streams.read(artifact.root, "prepare", {}, [scope])
    assert loaded["streams"]["cases"]["count"] == 1000
    assert [c["ordinal"] for c in streams.cases(artifact.root, loaded)] == list(range(1000))
    assert (artifact.root / "phase.json").stat().st_size < 4096
    with pytest.raises(Invalid, match="new phase attempt"):
        streams.Artifact(artifact.root, "prepare", {})


def test_cut_and_diagnostics_keep_full_enumeration_ordinals(tmp_path):
    from experiments.numerics_rows import dense_phases as dense
    assert hasattr(dense, "case_ranges"), "feature absent: full row ordinal ranges"
    groups = []
    for size, rank in ((1, 0), (2, 0), (2, 1)):
        cut = {"size": size, "rank": rank}
        units = [row.new_unit({"module": "A", "tp": cut, "m_config": {"M": m, "chunk_size": 1, "k_split": 1}}) for m in (1, 2)]
        module = {"roles": [{"cols": 2, "separation": {"recipes": [{"column": 0, "values": [1, -1, 2]}]}, "scale_index": []}]}
        groups.append(({"name": "A"}, module, cut, units))
    ranges = dense.case_ranges(groups)
    assert [ranges[("A", s, r, "gate")] for s, r in ((1, 0), (2, 0), (2, 1))] == [[0, 9], [9, 9], [18, 9]]
    assert [ranges[("A", s, r, "diagnostics")] for s, r in ((1, 0), (2, 0), (2, 1))] == [[27, 18], [45, 18], [63, 18]]
    for part, start in (("gate", 9), ("diagnostics", 45)):
        artifact = streams.Artifact(tmp_path / part, "prepare", {})
        artifact.document.update(ordinal_start=start, case_set=part)
        scope = streams.unit_scope(groups[1][3][0], "wire")
        case = artifact.add(scope, {"M": 1, "chunk_size": 1, "k_split": 1}, "dense", {}, category="diagnostic" if part == "diagnostics" else "sweep")
        assert case["ordinal"] == start and case["id"] == json_digest([scope, "dense", start])


@pytest.mark.parametrize("change", ("duplicate", "gap", "id", "bytes", "stream", "count", "configuration", "unfinished"))
def test_stream_refuses_changed_order_and_bytes(tmp_path, change):
    artifact = stream_feature(tmp_path)
    scope = streams.unit_scope(row.new_unit({"m_config": {"M": 1}}), "wire")
    artifact.document["scopes"] = [scope]
    case = artifact.add(scope, {"M": 1}, "dense", {"input": torch.ones(1)}, category="sweep")
    artifact.finish()
    streams.read(artifact.root, "prepare", {}, [scope])
    if change == "bytes":
        (artifact.root / case["tensors"]["input"]["file"]).write_bytes(b"bad")
    elif change == "stream":
        with (artifact.root / "cases.jsonl").open("a") as target:
            target.write("{}\n")
    elif change == "count":
        artifact.document["streams"]["cases"]["count"] += 1
    elif change == "unfinished":
        artifact.document["completion"] = False
    else:
        if change == "gap": case["ordinal"] += 1
        if change == "id": case["id"] = "other"
        if change == "configuration": case["configuration"] = {"M": 2}
        from tests.test_numerics_rows_phases import rewrite_stream
        rewrite_stream(artifact, "cases", [case, copy.deepcopy(case)] if change == "duplicate" else [case])
    if change in ("count", "unfinished"):
        (artifact.root / "phase.json").write_text(json.dumps(artifact.document))
    with pytest.raises(Invalid):
        streams.read(artifact.root, "prepare", {}, [scope])


def test_group_uses_k_processes_and_waits_for_all_exits(tmp_path, monkeypatch):
    import importlib.util
    assert importlib.util.find_spec("experiments.numerics_rows.capture_group") is not None, "feature absent: K capture processes"
    from experiments.numerics_rows import capture_group as group, gpu_time
    monkeypatch.setattr(gpu_time, "preflight", lambda: {"verdict": "FREE"})
    import threading
    barrier = threading.Barrier(3)
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        barrier.wait(timeout=10)
        assert "--capture-group-proof" in command
        return SimpleNamespace(returncode=3 if "failed" in command else 0)
    monkeypatch.setattr(group.subprocess, "run", child)
    units = [{"out": str(tmp_path / name), "command": ["python3", name, "--phase", "capture", "--out", str(tmp_path / name)]}
             for name in ("first", "failed", "last")]
    assert group.capture_group(3, units) == 3
    assert len(calls) == 3
    assert [json.loads((Path(u["out"]) / "exit.json").read_text())["exit_status"] for u in units] == [0, 3, 0]
    with pytest.raises(Invalid, match="K must equal"):
        group.capture_group(2, units)
