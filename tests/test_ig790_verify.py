"""Real caller reads must reject changed bytes and incomplete digest receipts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

import box_artifacts
from tessera.producer_plan import producer_projection

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def projected_source(tmp_path):
    stack = "model.layers.2.feed_forward.experts"
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "num_experts": 2, "hidden_size": 128, "moe_intermediate_size": 128}))
    shards = {}
    for expert in range(2):
        tensors = {f"{stack}.{expert}.{role}.weight": torch.zeros(128, 128, dtype=torch.bfloat16)
                   for role in ("w1", "w2", "w3")}
        name = f"model-{expert}.safetensors"
        save_file(tensors, str(source / name))
        shards[name] = tensors
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({stack: {"grid": "E4M3", "q256": 1024,
                                       "source_layout": "unpacked_per_expert"}}))
    projection = producer_projection(source, plan)
    projection_path = tmp_path / "projection.json"
    projection_path.write_text(json.dumps(projection))
    return source, projection, projection_path, stack, shards


def _verify(projected_source, tmp_path, omitted=None):
    source, _, projection_path, stack, _ = projected_source
    package = box_artifacts.skip_now("measurements", "ig790-pqbridge-pkg")
    (tmp_path / "prismaquant").symlink_to(package, target_is_directory=True)
    env = dict(os.environ, PRISMAQUANT_DEV_MODE="1",
               PYTHONPATH=os.pathsep.join(map(str, (ROOT / "src", ROOT / "tools", tmp_path))))
    # A separate process keeps the external caller's imports out of the test suite.
    program = """
import json, sys
from pathlib import Path
from prismaquant.tessera_calibration_cache import CaptureSourceAuthentication
from ig790_verify import verify_projected_units
if sys.argv[4]:
    original = CaptureSourceAuthentication.receipt
    def incomplete(self):
        receipt = original(self)
        receipt['verified_files'] = [row for row in receipt['verified_files']
                                     if row['name'] != sys.argv[4]]
        return receipt
    CaptureSourceAuthentication.receipt = incomplete
receipt = verify_projected_units(sys.argv[1], json.loads(Path(sys.argv[2]).read_text()), sys.argv[3])
print('RESULT', json.dumps(receipt))
"""
    return subprocess.run([sys.executable, "-c", program, str(source), str(projection_path),
                           stack, omitted or ""], env=env, cwd=ROOT,
                          capture_output=True, text=True, timeout=90)


def test_every_consumed_shard_has_an_independent_digest(projected_source, tmp_path):
    source, projection, _, _, shards = projected_source
    result = _verify(projected_source, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(next(line.removeprefix("RESULT ") for line in result.stdout.splitlines()
                              if line.startswith("RESULT ")))
    rows = {row["name"]: row for row in receipt["verified_files"]}
    assert set(shards) <= rows.keys()
    for name, tensors in shards.items():
        row = rows[name]
        assert row["sha256"] == projection["source"]["files"][name]
        assert row["sha256"] == hashlib.sha256((source / name).read_bytes()).hexdigest()
        assert row["sha256_source"] == "fresh_descriptor_sha256"
        assert row["bytes_hashed"] == (source / name).stat().st_size
        assert row["payload_reads"] == len(tensors)


def test_same_shape_changed_bytes_refuse(projected_source, tmp_path):
    source, _, _, _, shards = projected_source
    name, tensors = next(iter(shards.items()))
    size = (source / name).stat().st_size
    next(iter(tensors.values())).fill_(1)
    save_file(tensors, str(source / name))
    assert (source / name).stat().st_size == size
    result = _verify(projected_source, tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"RuntimeError: calibration source differs from census producer: {name}" in result.stderr


def test_omitted_consumed_shard_refuses(projected_source, tmp_path):
    result = _verify(projected_source, tmp_path, omitted="model-1.safetensors")
    assert result.returncode != 0, result.stdout + result.stderr
    assert "RuntimeError: caller verification omits consumed shards: model-1.safetensors" in result.stderr
