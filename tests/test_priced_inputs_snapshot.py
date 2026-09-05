"""External input expectations bind the producer's loaded owners, not paths."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tessera.errors import GrammarError
from tessera.export import ActivationSource

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "priced_inputs_exporter", ROOT / "experiments/export_tessera_serving.py")
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)
KEY = "unit.input_global_scale"
IDENTITY = {"text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64, "fit_tokens": 4}


def _snapshot(tmp_path, activation=None, scales=None, edit=None):
    block = {"schema": "tessera.priced_export_inputs.v1",
             "hessian_capture_sha256": activation.capture_sha256() if activation else None,
             "input_global_scales": {} if scales is None else scales}
    if edit:
        edit(block)
    raw = json.dumps({"priced_inputs": block}).encode()
    path = tmp_path / "build.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_loaded_capture_replacement_is_refused(tmp_path):
    activation = ActivationSource({"unit": torch.eye(4)}, dict(IDENTITY))
    path, digest = _snapshot(tmp_path, activation)
    replacement = ActivationSource({"unit": torch.eye(4) * 2}, dict(IDENTITY))
    with pytest.raises(SystemExit, match="Hessian capture differs"):
        exporter.PricedInputsSnapshot(path, digest).require(replacement, {})


def test_checked_capture_keeps_the_existing_consumption_seal(tmp_path):
    activation = ActivationSource({"unit": torch.eye(4)}, dict(IDENTITY), ldlq_sigma=None)
    path, digest = _snapshot(tmp_path, activation)
    exporter.PricedInputsSnapshot(path, digest).require(activation, {})
    activation.hessians["unit"].mul_(2)
    with pytest.raises(GrammarError, match="seal"):
        activation.for_unit("unit.weight", 4, "cpu")


@pytest.mark.parametrize("loaded", [{}, {KEY: 4.0}])
def test_missing_or_replaced_loaded_scale_refuses(tmp_path, loaded):
    path, digest = _snapshot(tmp_path, scales={KEY: 1.0})
    with pytest.raises(SystemExit, match="input_global_scale differs"):
        exporter.PricedInputsSnapshot(path, digest).require(None, loaded)


def test_replaced_build_cannot_change_the_expected_snapshot(tmp_path):
    path, digest = _snapshot(tmp_path, scales={KEY: 1.0})
    _snapshot(tmp_path, scales={KEY: 4.0})
    with pytest.raises(SystemExit, match="build SHA-256"):
        exporter.PricedInputsSnapshot(path, digest)


def test_build_file_is_not_read_again_after_intake(tmp_path):
    path, digest = _snapshot(tmp_path, scales={KEY: 1.0})
    snapshot = exporter.PricedInputsSnapshot(path, digest)
    path.write_text("{}")
    snapshot.require(None, {KEY: 1.0})


@pytest.mark.parametrize("expected_h,loaded_h", [(False, True), (True, False)])
def test_hessian_presence_is_part_of_the_price(tmp_path, expected_h, loaded_h):
    activation = ActivationSource({"unit": torch.eye(4)}, dict(IDENTITY))
    path, digest = _snapshot(tmp_path, activation if expected_h else None)
    with pytest.raises(SystemExit, match="Hessian capture differs"):
        exporter.PricedInputsSnapshot(path, digest).require(activation if loaded_h else None, {})


@pytest.mark.parametrize("edit", [
    lambda b: b.pop("hessian_capture_sha256"),
    lambda b: b.update(schema="unknown"),
    lambda b: b.update(hessian_capture_sha256="bad"),
    lambda b: b.update(input_global_scales={KEY: True}),
    lambda b: b.update(input_global_scales={KEY: float("nan")}),
    lambda b: b.update(input_global_scales={KEY: 0.0}),
])
def test_malformed_expectation_fails_closed(tmp_path, edit):
    path, digest = _snapshot(tmp_path, edit=edit)
    with pytest.raises(SystemExit, match="--priced-inputs"):
        exporter.PricedInputsSnapshot(path, digest)
