"""Shared priced-input documents and the production exporter binding."""
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


exporter = importlib.import_module("tessera.export_serving")


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
