"""Image identity drift stamps in development; unavailable images still refuse."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from tessera.serving.runtime_image import RuntimeImageError

PATH = Path(__file__).parents[1] / "experiments/t4_code/geometry_runtime_image.py"
PIN = "vllm/vllm-openai@sha256:" + "a" * 64
OTHER = "vllm/vllm-openai@sha256:" + "b" * 64
CONTRACT = {"versions": {"default_serve_image": PIN}}


def owner():
    spec = importlib.util.spec_from_file_location("geometry_runtime_image", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def present(_image):
    return {"present": True, "repo_digests": [OTHER], "local_id": "observed-local-image"}


@pytest.mark.parametrize("image", [PIN, "vllm/vllm-openai:local"])
def test_actual_identity_mismatch_stamps_then_returns_observed_record(image, monkeypatch, capsys):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    record = owner().measurement_image(image, inspector=present, contract=CONTRACT)
    assert record["refused"] is False
    assert record["resolved_reference"] == OTHER
    assert record["identity_refusal"] in ("image_digest_mismatch", "image_pin_mismatch")
    assert "[DEV-MODE]" in capsys.readouterr().out


@pytest.mark.parametrize("image", [PIN, "vllm/vllm-openai:local"])
def test_certified_identity_mismatch_keeps_refusal(image, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(RuntimeImageError, match="not the required serving image"):
        owner().measurement_image(image, inspector=present, contract=CONTRACT)


@pytest.mark.parametrize("mode", ["1", "0"])
def test_image_absence_is_not_identity_drift(mode, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    with pytest.raises(RuntimeImageError) as caught:
        owner().measurement_image(PIN, inspector=lambda _: {"present": False}, contract=CONTRACT)
    assert caught.value.payload["reason"] == "image_absent"


def test_image_owner_import_has_no_torch_or_shared_sdk_dependency():
    root = Path(__file__).parents[1]
    result = subprocess.run([sys.executable, "-S", "-c", "import importlib.util, sys; s=importlib.util.spec_from_file_location('g','experiments/t4_code/geometry_runtime_image.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert 'torch' not in sys.modules and 'prismabuild' not in sys.modules"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
