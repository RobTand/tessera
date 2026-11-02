"""Shared miniature checkpoints and selected-exporter fixture installs."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from tessera import export_serving as export
from tessera import source_profiles


ROOT = Path(__file__).resolve().parents[1]


HIDDEN, MOE_INTER, EXPERTS = 128, 256, 3


LAYER = "model.language_model.layers.1"


STACK = f"{LAYER}.mlp.experts"


PART_ARGV = ("--partition", "0/1", "--partition-runtime-image",
             "test/image@sha256:" + "a" * 64)


def _config(experts=EXPERTS):
    return {"architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {"hidden_size": HIDDEN, "moe_intermediate_size": MOE_INTER,
                            "num_hidden_layers": 2, "n_routed_experts": experts}}


def _checkpoint(experts=EXPERTS):
    """Unstacked 2-D expert projections -- the real census layout, one unit per name."""
    generator = torch.Generator().manual_seed(11)

    def normal(*shape):
        return torch.randn(*shape, generator=generator) * 0.02

    tensors = {
        "model.language_model.layers.0.mlp.gate_proj.weight": normal(2 * HIDDEN, HIDDEN),
        "model.language_model.layers.0.mlp.up_proj.weight": normal(2 * HIDDEN, HIDDEN),
        "model.language_model.layers.0.mlp.down_proj.weight": normal(HIDDEN, 2 * HIDDEN),
        f"{LAYER}.mlp.shared_experts.gate_proj.weight": normal(MOE_INTER, HIDDEN),
        f"{LAYER}.mlp.shared_experts.up_proj.weight": normal(MOE_INTER, HIDDEN),
        f"{LAYER}.mlp.shared_experts.down_proj.weight": normal(HIDDEN, MOE_INTER),
        f"{LAYER}.mlp.gate.weight": normal(experts, HIDDEN),
        "lm_head.weight": normal(HIDDEN, HIDDEN),
        "model.embed_tokens.weight": normal(HIDDEN, HIDDEN),
    }
    for index in range(experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shape = (HIDDEN, MOE_INTER) if projection == "down_proj" else (MOE_INTER, HIDDEN)
            tensors[f"{STACK}.{index}.{projection}.weight"] = normal(*shape)
    return tensors


def _write(tmp_path, tensors, config=None):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    safetensors_torch.save_file({k: v.contiguous() for k, v in tensors.items()},
                                str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps(config or _config()))
    return src


def _fixture_output_sizes(*, hidden=HIDDEN, mlp_inter=2 * HIDDEN, shared_inter=MOE_INTER):
    return {
        "language_model.model.layers.*.self_attn.o_proj": [hidden],
        "language_model.model.layers.*.mlp.down_proj": [hidden],
        "language_model.model.layers.*.mlp.gate_up_proj": [mlp_inter, mlp_inter],
        "language_model.model.layers.*.mlp.shared_experts.down_proj": [hidden],
        "language_model.model.layers.*.mlp.shared_experts.gate_up_proj": [shared_inter, shared_inter],
    }


def _declare_fixture_geometry(monkeypatch, *, hidden=HIDDEN,
                              mlp_inter=2 * HIDDEN, shared_inter=MOE_INTER):
    """Declare the small fixture partitions without a production contract change."""
    import copy
    from tessera.serving.contract import construction_entry as live_entry

    sizes = _fixture_output_sizes(hidden=hidden, mlp_inter=mlp_inter, shared_inter=shared_inter)

    def _entry(architectures, contract=None):
        entry = live_entry(architectures) if contract is None else live_entry(architectures, contract)
        if entry is None or entry.get("architecture") != "Glm5NextForConditionalGeneration":
            return entry
        entry = copy.deepcopy(entry)
        entry.setdefault("output_sizes", {}).update(sizes)
        return entry

    monkeypatch.setattr(export, "construction_entry", _entry)


def _git(*args, cwd, check=True):
    return subprocess.run(["git", *args], cwd=str(cwd), check=check,
                          capture_output=True, text=True)


def _fixture_install(target: Path) -> Path:
    """A synthetic site-packages: the package plus a dist-info naming it."""
    target.mkdir()
    config = source_profiles.packaged_projection_config((ROOT / "pyproject.toml").read_bytes())
    tracked = [p.relative_to(ROOT).as_posix() for p in (ROOT / "src" / "tessera").rglob("*")
               if p.is_file() and "__pycache__" not in p.parts]
    for name in source_profiles.shipped_payload_paths(config, tracked):
        original = ROOT / "src" / name
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    from tessera import DISTRIBUTION

    dist_info = target / f"{DISTRIBUTION.replace('-', '_')}-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {DISTRIBUTION}\nVersion: 0.1\n")
    return target


def _selected_export_argv(src, out, plan_path):
    return [str(src), str(out), "--device", "cpu",
            "--grid", "E4M3", "--q256", "1024", "--passthrough-unrouted",
            "--plan-json", str(plan_path), *PART_ARGV]


def _selected_export_env(site, repo):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(site)
    env["TESSERA_PRODUCER_PYTHON"] = sys.executable
    env["TESSERA_PRODUCER_SOURCE"] = str(repo / "src" / "tessera")
    env.pop("TESSERA_GIT", None)
    return env
