"""The export refuses a wire the pinned runtime will not route (#99).

RED FIRST.  On the tree before this gate, exporting a GLM checkpoint whose plan
includes attention succeeded: the modules are 2-D and under ``BODY_LAYER``, so
they were planned, encoded and declared -- and the runtime builds every one of
them with ``quant_config=None``, so the plugin is never asked, the wire is never
executed, and the ``<module>.weight`` it replaced is no longer in the
checkpoint.  Unlike #86's case that does not even end in a refusal: nothing on
either side can see the prefix.

So the gate is at EXPORT, before the first encode, and its input is the
contract's ``construction`` block rather than anything this script believes.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "export_tessera_serving", ROOT / "experiments" / "export_tessera_serving.py")
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)

BODY = "model.language_model.layers.0."
ROUTED = BODY + "mlp.down_proj.weight"
UNROUTED = BODY + "self_attn.o_proj.weight"


def _checkpoint(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    save_file({ROUTED: torch.randn(32, 64, dtype=torch.bfloat16),
               UNROUTED: torch.randn(32, 64, dtype=torch.bfloat16)},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps({
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {"hidden_size": 64, "moe_intermediate_size": 32},
    }))
    return src


def test_the_census_answers_before_anything_is_encoded():
    """The gate's own question, on the module names the exporter would declare."""
    config = {"architectures": ["Glm5NextForConditionalGeneration"]}
    verdicts, entry = exporter.unrouted_modules(
        config, [BODY + "mlp.down_proj", BODY + "self_attn.o_proj",
                 BODY + "self_attn.qkv_proj"])
    assert entry is not None
    assert verdicts[BODY + "self_attn.o_proj"][0] == "never_offered"
    assert verdicts[BODY + "self_attn.qkv_proj"][0] == "absent"
    assert BODY + "mlp.down_proj" not in verdicts


def test_an_uncensused_architecture_is_a_refusal_not_a_pass():
    verdicts, entry = exporter.unrouted_modules(
        {"architectures": ["NoSuchModelForCausalLM"]}, [BODY + "mlp.down_proj"])
    assert entry is None
    assert verdicts[BODY + "mlp.down_proj"][0] == "uncensused"


def test_export_refuses_and_names_the_prefixes(tmp_path, monkeypatch, capsys):
    src = _checkpoint(tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["export_tessera_serving.py", str(src), str(tmp_path / "out"),
                     "--grid", "E4M3", "--q256", "1024", "--device", "cpu", "--no-verify"])
    with pytest.raises(SystemExit) as excinfo:
        exporter.main()
    message = str(excinfo.value)
    assert "self_attn.o_proj" in message
    assert "never_offered" in message
    assert "quant_config=None" in message
    # And it refused BEFORE writing anything.
    assert not (tmp_path / "out" / "model.safetensors").exists()


def test_passthrough_unrouted_keeps_only_what_the_runtime_routes(tmp_path, monkeypatch):
    src = _checkpoint(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv", ["export_tessera_serving.py", str(src), str(out),
                     "--grid", "E4M3", "--q256", "1024", "--device", "cpu", "--no-verify",
                     "--passthrough-unrouted"])
    exporter.main()
    config = json.loads((out / "config.json").read_text())["quantization_config"]
    targets = sorted(t for group in config["config_groups"].values() for t in group["targets"])
    assert targets == [BODY + "mlp.down_proj"]
    # The unrouted Linear kept its own weight, which is the whole point.
    from safetensors import safe_open
    with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        keys = set(handle.keys())
    assert UNROUTED in keys
    assert ROUTED not in keys
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    gate = manifest["serving_gate"]
    assert gate["passthrough_unrouted"] is True
    assert [row["module"] for row in gate["unrouted"]] == [BODY + "self_attn.o_proj"]
    assert gate["construction_census"]["runtime"]["image"] == \
        "prismaquant/glm53-mia-sm121:487ecf187"


def test_allow_unrouted_writes_the_dead_wire_and_stamps_it(tmp_path, monkeypatch):
    """The research escape, and the artifact says so in its own bytes."""
    src = _checkpoint(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv", ["export_tessera_serving.py", str(src), str(out),
                     "--grid", "E4M3", "--q256", "1024", "--device", "cpu", "--no-verify",
                     "--allow-unrouted"])
    exporter.main()
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    gate = manifest["serving_gate"]
    assert gate["allow_unrouted"] is True
    assert [row["module"] for row in gate["unrouted"]] == [BODY + "self_attn.o_proj"]
    from safetensors import safe_open
    with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        keys = set(handle.keys())
    # The defect, reproduced deliberately: the weight is gone and a wire the
    # runtime will never execute stands in its place.
    assert UNROUTED not in keys
    assert BODY + "self_attn.o_proj.wire_bytes" in keys
