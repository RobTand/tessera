"""Cached-routed exports must build the supplement plan before either CLI path.

These probes cover plan construction, sealed-source preflight, and backend
command dispatch, not wire construction or GPU qualification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from child_env import child_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import pq2459_fixture_export as exporter
from tools import pq2459_source

PREFIX = "model.language_model.layers."
# This eleven-unit roster is the additive fixture decision, not a format menu.
DENSE_RECIPES = {
    PREFIX + suffix + ".weight": {"grid": grid, "q256": q256}
    for suffix, grid, q256 in (
        ("0.mlp.gate_proj", "BF16", 880),
        ("0.mlp.up_proj", "BF16", 880),
        ("0.mlp.down_proj", "BF16", 880),
        ("1.mlp.gate_proj", "BF16", 1088),
        ("1.mlp.up_proj", "BF16", 1088),
        ("1.mlp.down_proj", "BF16", 1024),
        ("2.mlp.gate_proj", "E4M3", 1024),
        ("2.mlp.up_proj", "E4M3", 1024),
        ("2.mlp.down_proj", "E4M3", 1088),
        ("3.mlp.shared_experts.gate_proj", "BF16", 880),
        ("3.mlp.shared_experts.up_proj", "BF16", 880),
    )
}
ROUTED = PREFIX + "4.mlp.experts"
UNSELECTED = PREFIX + "3.self_attn.q_proj.weight"


@pytest.fixture
def invocation(tmp_path):
    original = {name: {"grid": "E4M3", "q256": 960} for name in DENSE_RECIPES}
    original.update({ROUTED: {"grid": "BF16", "q256": 1024},
                     PREFIX + "5.mlp.experts": "PASSTHROUGH",
                     UNSELECTED: {"grid": "E4M3", "q256": 960}})
    original_path = tmp_path / "original.json"
    original_path.write_text(json.dumps(original))
    source = tmp_path / "source"
    source.mkdir()
    output, plan = tmp_path / "export", tmp_path / "plan.json"
    archive = ROOT / "experiments/pq2459/source-9eef9fea6.tar"
    cache = tmp_path / "cached-routed.json"
    # Backend cache intake is outside these preflight/dispatch probes.
    args = ["--source", str(source), "--original-plan", str(original_path),
            "--cached-routed", str(cache), "--producer-package", str(tmp_path / "producer"),
            "--producer-source-sha256", "dispatch-producer-digest",
            "--producer-authority", str(ROOT / "experiments/pq2459/producer_authority.py"),
            "--hessian", str(tmp_path / "hessian"),
            "--source-digest-cache", str(tmp_path / "digests"),
            "--output", str(output), "--plan-output", str(plan), "--intake-threads", "1",
            "--source-archive", str(archive),
            "--source-archive-sha256", hashlib.sha256(archive.read_bytes()).hexdigest()]
    return SimpleNamespace(args=args, original=original, original_path=original_path,
                           source=source, output=output, plan=plan, cache=cache)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_routed_preflight_uses_sealed_parsers(invocation, tmp_path, device):
    import torch
    from safetensors.torch import save_file

    names = [*DENSE_RECIPES, UNSELECTED,
             *(ROUTED + ".0." + role + ".weight"
               for role in ("gate_proj", "up_proj", "down_proj"))]
    save_file({name: torch.zeros((2, 2)) for name in names},
              str(invocation.source / "model.safetensors"))
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/pq2459_fixture_export.py"),
         *invocation.args, "--device", device, "--preflight"],
        env=child_env(TMPDIR=str(tmp_path)), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # The producer-authority loader logs its path before the JSON receipt.
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt["dense_units"] == 11
    assert receipt["source_shards"] == 1
    assert receipt["routed_source_units"] == 3
    assert receipt["qualified_cells"] == 0
    assert Path(receipt["source"]["archive"]) == Path(
        invocation.args[invocation.args.index("--source-archive") + 1])
    assert not invocation.plan.exists()
    assert not invocation.output.exists()


@pytest.mark.parametrize("device,backend_status", [("cpu", 0), ("cuda", 7)])
def test_cached_routed_export_writes_supplement_and_dispatches(
        invocation, monkeypatch, device, backend_status):
    monkeypatch.setattr(pq2459_source, "activate_source", lambda *_args: {})
    commands = []

    def run(command, *, check):
        assert check is False
        commands.append(command)
        return SimpleNamespace(returncode=backend_status)

    monkeypatch.setattr(exporter.subprocess, "run", run)
    original_bytes = invocation.original_path.read_bytes()
    assert exporter.main([*invocation.args, "--device", device]) == backend_status
    plan = json.loads(invocation.plan.read_text())
    assert plan == {**DENSE_RECIPES, ROUTED: invocation.original[ROUTED],
                    PREFIX + "5.mlp.experts": "PASSTHROUGH", UNSELECTED: "PASSTHROUGH"}
    assert invocation.original_path.read_bytes() == original_bytes
    assert len(commands) == 1
    command = commands[0]
    prefix = ([sys.executable, "-m", "tessera.export_serving"] if device == "cpu" else
              [sys.executable, str(ROOT / "tools/pq2459_export_progress.py")])
    assert command == [
        *prefix, str(invocation.source), str(invocation.output), "--device", device,
        "--plan-json", str(invocation.plan), "--cached-expert-units", str(invocation.cache),
        "--cached-producer-package", invocation.args[invocation.args.index("--producer-package") + 1],
        "--cached-producer-source-sha256", "dispatch-producer-digest",
        "--producer-authority", str(ROOT / "experiments/pq2459/producer_authority.py"),
        "--hessian", invocation.args[invocation.args.index("--hessian") + 1],
        "--cached-hessian-identity", "committed", "--cached-intake-threads", "1",
        "--source-digest-cache", invocation.args[invocation.args.index("--source-digest-cache") + 1],
        "--fit-tp-size", "2",
    ]
    assert not invocation.output.exists()
