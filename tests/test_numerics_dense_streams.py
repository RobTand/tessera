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


def test_evidence_copy_consumes_one_file_name_at_a_time(tmp_path, monkeypatch):
    artifact = stream_feature(tmp_path)
    from experiments.numerics_rows.dense_phases import bind_evidence
    source = tmp_path / "compiled-code"
    source.mkdir()
    for name in ("first", "second"):
        (source / name).write_text(name)
    def evidence_paths(directory, pattern):
        assert directory == source
        yield source / "first"
        assert (artifact.root / "evidence/compiled-code/first").is_file()
        yield source / "second"
    monkeypatch.setattr(Path, "rglob", evidence_paths)
    bind_evidence(tmp_path, artifact)
    assert len(list(streams.records(artifact.root, "files"))) == 2


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


@pytest.mark.parametrize("change", ("duplicate", "overlap", "gap", "id", "bytes", "stream", "count", "configuration", "unfinished"))
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
        if change == "overlap":
            case["tensors"] = {}
            artifact.document["ranges"] = [[0, 1], [0, 1]]
        from tests.test_numerics_rows_phases import rewrite_stream
        rewrite_stream(artifact, "cases", [case, copy.deepcopy(case)] if change in ("duplicate", "overlap") else [case])
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


def test_stream_reader_checks_the_bytes_it_consumes(tmp_path):
    artifact = stream_feature(tmp_path)
    scope = streams.unit_scope(row.new_unit({"m_config": {"M": 1}}), "wire")
    artifact.document["scopes"] = [scope]
    case = artifact.add(scope, {"M": 1}, "dense", {}, category="sweep")
    artifact.finish()
    loaded, _ = streams.read(artifact.root, "prepare", {}, [scope])
    case["category"] = "diagnostic"
    (artifact.root / "cases.jsonl").write_text(json.dumps(case) + "\n")
    with pytest.raises(Invalid, match="stream digest"):
        list(streams.cases(artifact.root, loaded))


@pytest.mark.parametrize("tail", (["--phase"], ["--phase", "compare"], ["--phase", "capture", "--capture-group-proof", "other"]))
def test_group_validates_all_commands_before_any_process(tmp_path, monkeypatch, tail):
    import importlib.util
    assert importlib.util.find_spec("experiments.numerics_rows.capture_group") is not None, "feature absent: K capture processes"
    from experiments.numerics_rows import capture_group as group, gpu_time
    calls = []
    monkeypatch.setattr(gpu_time, "preflight", lambda: calls.append("preflight"))
    monkeypatch.setattr(group.subprocess, "run", lambda *a, **k: calls.append("child"))
    units = [{"out": str(tmp_path / name), "command": ["python3", "--out", str(tmp_path / name)] + flags}
             for name, flags in (("valid", ["--phase", "capture"]), ("bad", tail))]
    with pytest.raises(Invalid, match="capture phase"):
        group.capture_group(2, units)
    assert calls == [] and not any(Path(item["out"]).exists() for item in units)


@pytest.mark.parametrize("mutation", (None, "output", "array-index", "extra-array"))
def test_streamed_scale_arrays_keep_mismatch_counts_and_scope(tmp_path, monkeypatch, mutation):
    stream_feature(tmp_path)
    from experiments.numerics_rows import dense_phases as dense
    from experiments.numerics_rows.gpu_time import PhaseTimes
    unit = row.new_unit({"cell": "cell", "module": "A", "role": "unit", "arm": {},
                         "m_config": {"M": 1}, "tp": {"size": 1, "rank": 0}})
    unit["exact"]["separation"]["records"] = [{"both": 1}]
    unit["exact"]["sweep"]["elements"] = unit["exact"]["chunk_pairs"]["x_elements"] = 1
    unit["exact"]["scale_index"].update(streamed=True, launches=1)
    row.finish_unit(unit)
    receipt = {"row": {"utc_start": "start", "finalization_finished": False}, "units": [unit], "runtime": {}, "modules": []}
    monkeypatch.setattr(dense, "identities", lambda record: {})
    scope = streams.unit_scope(unit, "wire")
    prepared = streams.Artifact(tmp_path / "source-prepare", "prepare", {})
    prepared.document["scopes"] = [scope]
    x, scale = torch.ones(1, 1, dtype=torch.bfloat16), torch.ones(1, 1)
    base = {"input": x, "q": x, "sf": scale, "reference": x * 2,
            "finite_value": scale * 10, "finite_held": scale.bool(), "x_mask": scale.bool()}
    prepared.add(scope, {"M": 1}, "dense", base, category="sweep")
    prepared.add(scope, {"M": 1}, "scale", {**base, "reference": (x * 2).reshape(1)},
                 category="scale_index", array_index=int(mutation == "array-index"), output_index=[0, 0])
    array = {"scope": scope, "array_index": 0, "record": {"scale": "row", "base": 2, "families": [],
             "passes_total": 1, "reads": 1, "reads_covered": 1, "launches": 1, "mismatches": 0}}
    prepared.record("arrays", array)
    if mutation == "extra-array": prepared.record("arrays", {**array, "array_index": 1})
    digest = prepared.finish(receipt=copy.deepcopy(receipt))
    captured = streams.Artifact(tmp_path / "source-capture", "capture", {})
    captured.document.update(scopes=[scope], prepare_digest=digest)
    for kind in ("dense", "scale"):
        output = x * (3 if mutation == "output" and kind == "scale" else 2)
        captured.add(scope, {"M": 1}, kind, {"output": output, "staged": output, "quantizer_q": x, "quantizer_sf": scale},
                     category="sweep" if kind == "dense" else "scale_index", completion=True,
                     boundaries=[{"kind": "quantizer", "sync_before": True, "sync_after": True}] * 2,
                     profiler_names=["kernel"], kernel_origins=[{"profiler_name": "kernel"}])
    captured.finish(receipt=copy.deepcopy(receipt))
    out = tmp_path / "compare-job"; out.mkdir()
    arguments = SimpleNamespace(out=out, prepare_artifact=[prepared.root], capture_artifact=[captured.root])
    groups = [({"name": "A"}, {"wire_sha256": "wire"}, unit["tp"], [unit])]
    if mutation in ("array-index", "extra-array"):
        with pytest.raises(Invalid, match="ordered array|extra scale arrays"):
            dense.host_compare(arguments, receipt, PhaseTimes({}), groups)
        assert not receipt["row"]["finalization_finished"]
    else:
        dense.host_compare(arguments, receipt, PhaseTimes({}), groups)
        final = receipt["units"][0]
        assert receipt["row"]["finalization_finished"] and final["outcome"] == ("FAIL" if mutation else "PASS")
        assert final["exact"]["scale_index"]["mismatches"] == int(mutation == "output")
        assert final["exact"]["scale_index"]["arrays"] == []
