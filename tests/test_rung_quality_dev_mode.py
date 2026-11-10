"""Path identity stamps never replace numerical quality validity checks."""
import importlib.util
import json
from pathlib import Path
import shutil
import hashlib
import subprocess

import pytest

from tessera.control import grid_for_name, unit_wire_bits


@pytest.fixture
def quality_binding(tmp_path, monkeypatch):
    module_path = Path(__file__).parents[1] / "experiments/t8r_speed/rung_quality.py"
    spec = importlib.util.spec_from_file_location("quality_binding_dev_mode", module_path)
    quality = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quality)
    cas = tmp_path / "requests"
    monkeypatch.setattr(quality, "RECORDED_REQUESTS", cas)
    action = "a" * 64
    original = tmp_path / "recorded" / "quality.json"
    original.parent.mkdir()
    bits = unit_wire_bits(grid_for_name("E4M3"), 1024, 32, 256)
    document = {"rungs": {"1024": {"samples": [{
        "exact_bytes": int(bits / 8),
        "accounted_bits": {"numerator": bits.numerator, "denominator": bits.denominator},
    }]}}}
    original.write_text(json.dumps(document))
    request = cas / f"{action}.json"
    request.parent.mkdir(parents=True)
    params = {
        "command": ["python", "experiments/t8r_speed/rung_quality.py", "--grid", "E4M3",
                    "--cases", "1024", "--out", str(original)],
        "checkout_snapshot": {"commit": "recorded-source"},
    }
    source = tmp_path / 'source-request.json'
    source.write_text(json.dumps({'params': params}))
    request.write_text(json.dumps({'action_key': action, 'provenance': {
        'source': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}, 'params': params}))
    _commit_requests(tmp_path)
    return quality, action, original, tmp_path / "scoped.json"


def _commit_requests(root):
    subprocess.run(['git', 'init', str(root)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(root), 'add', 'requests', 'source-request.json'], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-m', 'record requests'], check=True, capture_output=True)


def test_relocated_stored_quality_continues_in_default_dev_mode(quality_binding, monkeypatch, capsys):
    quality, action, original, output = quality_binding
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    relocated = original.parent.parent / "relocated.json"
    shutil.copyfile(original, relocated)
    original.unlink()
    quality.bind_existing_quality(relocated, action, output)
    scope = json.loads(output.read_text())["rungs"]["1024"]["scope"]
    assert scope["format"] == "TESSERA_E4M3_K1"
    assert scope["grid"] == "E4M3"
    assert scope["arity"] == 1
    assert scope["rung"] == 1024
    assert "[DEV-MODE]" in capsys.readouterr().out


def test_relocated_quality_identity_refuses_in_certified_mode(quality_binding, monkeypatch):
    quality, action, original, output = quality_binding
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    relocated = original.parent.parent / "relocated.json"
    shutil.copyfile(original, relocated)
    with pytest.raises(ValueError):
        quality.bind_existing_quality(relocated, action, output)


@pytest.mark.parametrize("mode", [None, "0", "1"])
def test_corrupt_encoder_byte_accounting_refuses_in_every_mode(quality_binding, monkeypatch, mode):
    quality, action, original, output = quality_binding
    if mode is None:
        monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    else:
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    document = json.loads(original.read_text())
    document["rungs"]["1024"]["samples"][0]["exact_bytes"] += 1
    original.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        quality.bind_existing_quality(original, action, output)


@pytest.mark.parametrize("fault", ["absent", "no-provenance"])
def test_recorded_request_must_exist_and_carry_its_provenance(quality_binding, fault):
    quality, action, original, output = quality_binding
    request = Path(quality.RECORDED_REQUESTS) / f"{action}.json"
    quality.bind_existing_quality(original, action, output)  # the accepted control
    output.unlink()
    if fault == "absent":
        request.unlink()
        error = FileNotFoundError
    else:
        value = json.loads(request.read_text())
        del value["provenance"]
        request.write_text(json.dumps(value))
        error = ValueError
    with pytest.raises(error):
        quality.bind_existing_quality(original, action, output)


@pytest.mark.parametrize('action', ['../escape', 'A' * 64, 'a' * 63, 'g' * 64])
def test_quality_action_key_has_no_path_grammar(quality_binding, action):
    quality, _, original, output = quality_binding
    with pytest.raises(ValueError, match='action_key.*64 lowercase hex'):
        quality.bind_existing_quality(original, action, output)


def test_committed_claimed_hash_is_compared_to_original_bytes(quality_binding):
    quality, action, original, output = quality_binding
    quality.bind_existing_quality(original, action, output)
    output.unlink()
    request = quality.RECORDED_REQUESTS / (action + '.json')
    record = json.loads(request.read_text())
    record['provenance']['sha256'] = '0' * 64
    request.write_text(json.dumps(record))
    _commit_requests(request.parent.parent)
    with pytest.raises(ValueError, match='sha256 differs'):
        quality.bind_existing_quality(original, action, output)


def test_missing_recorded_requests_still_fail_closed(quality_binding, tmp_path):
    quality, action, original, output = quality_binding
    with pytest.raises(FileNotFoundError):
        quality.bind_existing_quality(original, action, output, requests=tmp_path / 'absent')
