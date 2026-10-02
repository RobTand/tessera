"""Finite PM admission must not broaden the historical replay or trust a screen."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/t8r_speed/piece_major_protocol.py"
SPEC = importlib.util.spec_from_file_location("piece_major_protocol", PATH)
pp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pp)


def protocol(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    item = {"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    return {"schema": pp.SCHEMA, "artifact": pp.ARTIFACT, "group": pp.GROUP,
            "ms": [1, 2048], "order": pp.ORDER.copy(), "warmup": 10, "iters": 30, "power_s": 30,
            "kernel_sha256": "a" * 64, "harness": {"experiments/t8r_speed/" + name: "a" * 64
                for name in ("bench_t8r.py", "bench_t8r.sh", "piece_major_protocol.py", "pb_staged_store.py")}, "input_manifest": item,
            "source_manifest": item.copy(), "native": {**item, "origin_path": str(manifest),
                "files": {"fixture.so": {"sha256": "a" * 64, "bytes": 1, "mtime_ns": 0}}}, "routing": item.copy()}


@pytest.mark.parametrize("field,value", [
    ("artifact", "/different"), ("group", "experts.R1024.L11"), ("ms", [1, 512, 2048]),
    ("order", ["legacy", "piece_major"]), ("warmup", 11), ("iters", 31), ("power_s", 31),
    ("schema", "future"), ("kernel_sha256", "a" * 63),
])
def test_protocol_refuses_scope_or_budget_change(tmp_path, field, value):
    doc = protocol(tmp_path)
    doc[field] = value
    with pytest.raises(ValueError):
        pp.validate(doc)


def test_protocol_refuses_unknown_extension(tmp_path):
    doc = protocol(tmp_path)
    doc["retry"] = True
    with pytest.raises(ValueError):
        pp.validate(doc)


def options(doc):
    return SimpleNamespace(artifact=doc["artifact"], groups=doc["group"], ms="1,2048",
        no_graph=True, outputs_only=False, routing=None, single_routing_file=None,
        profile_native_file=None, input_manifest=doc["input_manifest"]["path"],
        warmup=10, iters=30, power_s=30, comparison_phase="numeric", ncu=False)


def environment(monkeypatch, doc):
    for key in ("TESSERA_ROUTED_FUSED", "TESSERA_FUSED_E4M3_MMA", "TESSERA_ROUTED_FUSED_WIDE"):
        monkeypatch.setenv(key, "e4m3" if key == "TESSERA_FUSED_E4M3_MMA" else "1")
    monkeypatch.setenv("BENCH_EXPECT_LIBRARY_SHA256", doc["native"]["sha256"])


def test_exact_numeric_arguments_and_manifest_admit(tmp_path, monkeypatch):
    doc = protocol(tmp_path)
    environment(monkeypatch, doc)
    pp.require_options(options(doc), pp.validate(doc))


@pytest.mark.parametrize("field,value", [
    ("no_graph", False), ("outputs_only", True), ("routing", "foreign"),
    ("single_routing_file", "foreign"), ("profile_native_file", "foreign"),
    ("comparison_phase", "unbounded"), ("ncu", True), ("input_manifest", "foreign"),
])
def test_actual_options_refuse_alternate_execution(tmp_path, monkeypatch, field, value):
    doc = protocol(tmp_path)
    environment(monkeypatch, doc)
    args = options(doc)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        pp.require_options(args, doc)


def test_manifest_drift_and_stubbed_vllm_refuse(tmp_path, monkeypatch):
    doc = protocol(tmp_path)
    environment(monkeypatch, doc)
    with pytest.raises(ValueError, match="stubbed"):
        pp.require_options(options(doc), doc, stubbed=True)
    Path(doc["input_manifest"]["path"]).write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="digest differs"):
        pp.require_options(options(doc), doc)


def receipt(tmp_path, **overrides):
    doc = {"schema": pp.SCHEMA + ".receipt", "phase": "numeric", "status": "passed",
           "protocol_sha256": "b" * 64, "ms": [1, 2048], "intermediate_bits_equal": True,
           "source_files_unchanged": True,
           "results": [{"M": m, "intermediate_bits_equal": True,
                        "bit_hashes": {key: "a" * 64 for key in ("forward", "mode0", "mode1", "mode2")},
                        "input_hashes": {key: "a" * 64 for key in ("x", "ids", "weights")}}
                       for m in (1, 2048)]}
    doc.update(overrides)
    path = tmp_path / "numeric.json"
    path.write_text(json.dumps(doc))
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def test_numeric_receipt_is_content_and_protocol_bound(tmp_path):
    path, digest = receipt(tmp_path)
    pp.require_numeric_receipt(path, digest, "b" * 64)
    with pytest.raises(ValueError):
        pp.require_numeric_receipt(path, digest, "c" * 64)
    Path(path).write_text("{}")
    with pytest.raises(ValueError, match="digest differs"):
        pp.require_numeric_receipt(path, digest, "b" * 64)


@pytest.mark.parametrize("change", [
    {"phase": "timing"}, {"status": "failed"}, {"ms": [2048]},
    {"intermediate_bits_equal": False}, {"schema": "output_screen"}, {"results": []},
    {"intermediate_bits_equal": "true"}, {"source_files_unchanged": False},
])
def test_a_screen_or_incomplete_numeric_result_cannot_admit_timing(tmp_path, change):
    path, digest = receipt(tmp_path, **change)
    with pytest.raises(ValueError):
        pp.require_numeric_receipt(path, digest, "b" * 64)


def test_timing_without_numeric_receipt_refuses():
    with pytest.raises(ValueError):
        pp.require_numeric_receipt(None, None, "b" * 64)


def test_cold_script_resolves_protocol_owner_without_checkout_cwd(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import os
    import subprocess
    import sys
    root = PATH.parents[2]
    env = dict(os.environ, PYTHONPATH=str(root / 'src'), CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, str(root / 'experiments/t8r_speed/bench_t8r.py'),
                             '--out', str(tmp_path / 'out'), '--comparison-protocol',
                             str(tmp_path / 'missing.json')], cwd=tmp_path, env=env,
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert 'FileNotFoundError' in result.stderr, result.stderr
    assert "No module named 'experiments'" not in result.stderr


@pytest.mark.parametrize('choice,accepted', [('e4m3', True), ('1', False), ('f16', False)])
def test_comparison_admission_matches_actual_mma_selector(tmp_path, monkeypatch, choice, accepted):
    pytest.importorskip('torch')
    from tessera import routed_fused as rf
    doc = protocol(tmp_path)
    environment(monkeypatch, doc)
    monkeypatch.setenv(rf.ENV_E4M3_MMA, choice)
    if accepted:
        assert rf.library_for('e4m3') == 'e4m3mma'
        pp.require_options(options(doc), doc)
    else:
        with pytest.raises(ValueError):
            pp.require_options(options(doc), doc)
