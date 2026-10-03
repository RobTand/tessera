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


@pytest.mark.parametrize('change', [None, 'kernel_sha256', 'routing', 'native', 'source_manifest'])
def test_timing_reuses_only_semantically_identical_numeric_protocol(tmp_path, change):
    original = protocol(tmp_path)
    path = tmp_path / 'numeric-protocol.json'
    path.write_text(json.dumps(original))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    timing = copy.deepcopy(original)
    timing['harness'] = {name: 'c' * 64 for name in original['harness']}
    timing['numeric_protocol'] = {'path': str(path), 'sha256': digest}
    if change:
        if change == 'kernel_sha256': timing[change] = 'd' * 64
        else: timing[change]['sha256'] = 'd' * 64
        with pytest.raises(ValueError, match='semantic input differs'):
            pp.numeric_protocol_sha256(timing, 'e' * 64)
    else:
        assert pp.numeric_protocol_sha256(pp.validate(timing), 'e' * 64) == digest

REPEAT_SCHEMA = "tessera.routed_piece_major_repeatability.v1"


def repeatability_protocol(tmp_path, pair=1, block=1):
    original = protocol(tmp_path)
    original_path = tmp_path / 'original-numeric-protocol.json'
    original_path.write_text(json.dumps(original))
    doc = copy.deepcopy(original)
    doc.update(schema=REPEAT_SCHEMA, ms=[2048], power_s=0,
               order=pp.ORDER.copy() if (pair, block) in ((1, 1), (2, 2), (3, 1)) else ['piece_major', 'legacy', 'legacy', 'piece_major'])
    doc['numeric_protocol'] = {'path': str(original_path),
        'sha256': hashlib.sha256(original_path.read_bytes()).hexdigest()}
    doc['repeatability'] = {'pair': pair, 'block': block, 'conditioning_s': 60,
        'profile_receipt': {'path': str(tmp_path / 'accepted-profiles.json'), 'sha256': 'f' * 64}}
    return doc


@pytest.mark.parametrize('pair,block', [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)])
def test_repeatability_admits_only_predeclared_balanced_block(tmp_path, pair, block):
    doc = repeatability_protocol(tmp_path, pair, block)
    assert pp.validate(doc) is doc
    assert pp.numeric_protocol_sha256(doc, 'e' * 64) == doc['numeric_protocol']['sha256']


@pytest.mark.parametrize('field,value', [('ms',[1,2048]), ('power_s',30),
    ('order',['legacy','piece_major']), ('warmup',11), ('iters',31)])
def test_repeatability_cannot_broaden_control_population(tmp_path, field, value):
    doc = repeatability_protocol(tmp_path)
    doc[field] = value
    with pytest.raises(ValueError):
        pp.validate(doc)


@pytest.mark.parametrize('field,value', [('pair',True), ('pair',4), ('block',0),
    ('conditioning_s',59), ('conditioning_s',61)])
def test_repeatability_requires_exact_identity_and_conditioning(tmp_path, field, value):
    doc = repeatability_protocol(tmp_path)
    doc['repeatability'][field] = value
    with pytest.raises(ValueError):
        pp.validate(doc)


def test_repeatability_still_binds_numeric_semantics_and_actual_options(tmp_path, monkeypatch):
    doc = repeatability_protocol(tmp_path)
    environment(monkeypatch, doc)
    args = options(doc)
    args.ms, args.power_s, args.comparison_phase = '2048', 0, 'repeatability'
    pp.require_options(args, pp.validate(doc))
    for phase in ('numeric', 'timing', 'ncu'):
        args.comparison_phase = phase
        with pytest.raises(ValueError):
            pp.require_options(args, doc)
    doc['routing']['sha256'] = 'd' * 64
    with pytest.raises(ValueError, match='semantic input differs'):
        pp.numeric_protocol_sha256(doc, 'e' * 64)


def test_old_protocol_cannot_admit_repeatability(tmp_path, monkeypatch):
    doc = protocol(tmp_path)
    environment(monkeypatch, doc)
    args = options(doc)
    args.comparison_phase = 'repeatability'
    with pytest.raises(ValueError):
        pp.require_options(args, doc)


def test_repeatability_requires_content_bound_unchanged_profile_population(tmp_path):
    doc = repeatability_protocol(tmp_path)
    numeric_path, numeric_digest = receipt(tmp_path)
    numeric = pp.require_numeric_receipt(numeric_path, numeric_digest, 'b' * 64)
    profile = {'schema': pp.SCHEMA + '.receipt', 'phase': 'timing', 'status': 'passed',
        'ms': [1,2048], 'kernel_sha256': doc['kernel_sha256'], 'source_files_unchanged': True,
        'native': {arm: {'sha256':doc['native']['sha256'], 'files':doc['native']['files']}
                   for arm in ('legacy','piece_major')},
        'results': [{'M':m, 'input_hashes':numeric['results'][i]['input_hashes'],
            'cells': {f'{position}:{arm}:M{m}': {'profile':{'trace_sha256':'a'*64, 'top':{'kernel':{}}}}
                for position,arm in enumerate(pp.ORDER)}} for i,m in enumerate((1,2048))]}
    path = Path(doc['repeatability']['profile_receipt']['path'])
    path.write_text(json.dumps(profile))
    doc['repeatability']['profile_receipt']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert pp.require_profile_receipt(doc, numeric)['phase'] == 'timing'
    profile['native']['piece_major']['sha256'] = 'd' * 64
    path.write_text(json.dumps(profile))
    doc['repeatability']['profile_receipt']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='profile'):
        pp.require_profile_receipt(doc, numeric)
