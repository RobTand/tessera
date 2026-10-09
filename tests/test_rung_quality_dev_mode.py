"""Path identity stamps never replace numerical quality validity checks."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from tessera.control import grid_for_name, unit_wire_bits


@pytest.fixture
def quality_binding(tmp_path, monkeypatch):
    module_path = Path(__file__).parents[1] / "experiments/t8r_speed/rung_quality.py"
    spec = importlib.util.spec_from_file_location("quality_binding_dev_mode", module_path)
    quality = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quality)
    cas = tmp_path / "requests"
    real_path = quality.Path
    monkeypatch.setattr(
        quality, "Path",
        # The one production call is the PrismaBuild CAS request directory.  Match it by
        # its tail so the test never names a path on a box (tests/test_box_artifacts.py).
        lambda path: cas if str(path).endswith("/cas/requests") else real_path(path),
    )
    action = "a" * 64
    original = tmp_path / "recorded" / "quality.json"
    original.parent.mkdir()
    bits = unit_wire_bits(grid_for_name("E4M3"), 1024, 32, 256)
    document = {"rungs": {"1024": {"samples": [{
        "exact_bytes": int(bits / 8),
        "accounted_bits": {"numerator": bits.numerator, "denominator": bits.denominator},
    }]}}}
    original.write_text(json.dumps(document))
    request = cas / action[:2] / f"{action}.json"
    request.parent.mkdir(parents=True)
    request.write_text(json.dumps({"action_key": action, "params": {
        "command": ["python", "experiments/t8r_speed/rung_quality.py", "--grid", "E4M3",
                    "--cases", "1024", "--out", str(original)],
        "checkout_snapshot": {"commit": "recorded-source"},
    }}))
    return quality, action, original, tmp_path / "scoped.json"


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
