"""Every Linear the checkpoint carries is named, not only the ones in the body.

``ignore`` is not bookkeeping.  The plugin refuses a ``LinearBase`` that the
checkpoint neither declares nor ignores
(``serving/config.py::get_quant_method``), so a Linear the exporter copies
through at source precision and does not name is a checkpoint that loads only
where the runtime happens not to ask about that module.

The exporter's three ``ignore`` sources were all ``BODY_LAYER``-gated -- the
seed pair, the shard loop's ``if name in passthrough``, and the two expert
loops -- and ``BODY_LAYER`` matches ``model.<...>.layers.<N>.`` only.  A
vision tower (``model.visual.blocks.N.``) or an MTP sidecar (``mtp.layers.N.``)
is passed through and never named.  Read off
``/mnt/shared/models/GLM-5.3-Flash``: 24 blocks of ``attn.qkv``, ``attn.proj``
and ``mlp.{gate,up,down}_proj``, plus ``merger.{proj,gate_proj,up_proj,
down_proj}``.

WHAT THE RULE IS.  The name comes from the tensor that was WRITTEN, not from a
second exclusion list beside the first: a roster is a second place to remember,
and it goes stale in silence.  Every 2-D ``.weight`` copied through names the
vLLM module it belongs to -- its FUSED parent where vLLM merges it, because
vLLM builds one method per fused module (``Glm5NextVisionMLP`` builds one
``MergedColumnParallelLinear`` at ``{prefix}.gate_up_proj``, pinned build
``prismaquant/glm53-mia-sm121:487ecf187``, ``models/glm5next/nvidia/
multimodal.py:102-107``).

WHAT THIS DOES NOT CLAIM.  On that pinned build the vision tower is built with
``quant_config=None`` (``model.py:1082``), and ``LinearBase.__init__`` then
installs ``UnquantizedLinearMethod`` WITHOUT calling ``get_quant_method``
(``linear.py``, the ``if quant_config is None`` branch), so the plugin is never
asked about it and the refusal does not fire there today.  The rule is the
completeness the plugin's contract states, held for every Linear rather than
for the ones one runtime happens to ask about.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

_spec = importlib.util.spec_from_file_location(
    "export_tessera_serving",
    Path(__file__).resolve().parents[1] / "experiments" / "export_tessera_serving.py")
export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export)

HIDDEN, VIS = 128, 64


def _config():
    return {"architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {"hidden_size": HIDDEN, "moe_intermediate_size": 64,
                            "num_hidden_layers": 1},
            "vision_config": {"hidden_size": VIS, "depth": 1}}


def _checkpoint():
    """A body layer plus a vision tower, shaped like GLM-5.3-Flash's."""
    t = {}
    P = "model.language_model.layers.0"
    t[f"{P}.self_attn.q_proj.weight"] = torch.zeros(HIDDEN, HIDDEN)
    t[f"{P}.self_attn.k_proj.weight"] = torch.zeros(HIDDEN, HIDDEN)
    t[f"{P}.self_attn.v_proj.weight"] = torch.zeros(HIDDEN, HIDDEN)
    t[f"{P}.self_attn.o_proj.weight"] = torch.zeros(HIDDEN, HIDDEN)
    t[f"{P}.mlp.gate_proj.weight"] = torch.zeros(2 * HIDDEN, HIDDEN)
    t[f"{P}.mlp.up_proj.weight"] = torch.zeros(2 * HIDDEN, HIDDEN)
    t[f"{P}.mlp.down_proj.weight"] = torch.zeros(HIDDEN, 2 * HIDDEN)
    t["model.language_model.embed_tokens.weight"] = torch.zeros(256, HIDDEN)
    t["lm_head.weight"] = torch.zeros(256, HIDDEN)
    # The vision tower, outside BODY_LAYER.  Names and roles as on disk.
    V = "model.visual.blocks.0"
    t[f"{V}.attn.qkv.weight"] = torch.zeros(3 * VIS, VIS)
    t[f"{V}.attn.proj.weight"] = torch.zeros(VIS, VIS)
    t[f"{V}.mlp.gate_proj.weight"] = torch.zeros(2 * VIS, VIS)
    t[f"{V}.mlp.up_proj.weight"] = torch.zeros(2 * VIS, VIS)
    t[f"{V}.mlp.down_proj.weight"] = torch.zeros(VIS, 2 * VIS)
    t[f"{V}.norm1.weight"] = torch.zeros(VIS)
    t["model.visual.merger.proj.weight"] = torch.zeros(HIDDEN, VIS)
    t["model.visual.merger.gate_proj.weight"] = torch.zeros(VIS, HIDDEN)
    t["model.visual.merger.up_proj.weight"] = torch.zeros(VIS, HIDDEN)
    t["model.visual.merger.down_proj.weight"] = torch.zeros(HIDDEN, VIS)
    # A conv, rank 5: not a Linear, and no ignore entry belongs to it.
    t["model.visual.patch_embed.proj.weight"] = torch.zeros(VIS, 3, 2, 2, 2)
    return t


def _write(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    safetensors_torch.save_file({k: v.contiguous() for k, v in _checkpoint().items()},
                                str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps(_config()))
    return src


def _export(tmp_path, monkeypatch, *extra):
    src = _write(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out),
                                     "--grid", "E4M3", "--q256", "1024", *extra])
    export.main()
    return json.loads((out / "config.json").read_text())["quantization_config"]


#: The vision tower's Linears, spelled as vLLM builds them: the block MLP's
#: gate/up merge into one ``gate_up_proj``, the merger's likewise
#: (``multimodal.py:102-107``, ``:322-327``), and ``attn.qkv`` is already
#: merged on disk.
VISION_MODULES = (
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.attn.qkv_proj",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.mlp.gate_up_proj",
    "model.visual.blocks.0.mlp.down_proj",
    "model.visual.merger.proj",
    "model.visual.merger.gate_up_proj",
    "model.visual.merger.down_proj",
)


def test_the_ignore_rule_names_a_non_body_linear():
    """The rule itself, before any export: a passed-through Linear has a name."""
    # Both spellings: the pinned build names the merged module ``qkv_proj``
    # when it is given a quant config and ``qkv`` when it is not
    # (multimodal.py:167), and which one exists is not the producer's to know.
    assert export.ignored_modules("model.visual.blocks.0.attn.qkv.weight", (192, 64)) == (
        "model.visual.blocks.0.attn.qkv", "model.visual.blocks.0.attn.qkv_proj")
    assert export.ignored_modules("model.visual.blocks.0.mlp.gate_proj.weight", (128, 64)) == (
        "model.visual.blocks.0.mlp.gate_up_proj",)
    # The FUSED table's output, not an attested fact about Qwen4Exp: that
    # architecture is in no runtime on this box.  What it fixes is that the
    # rule reaches a name BODY_LAYER never matched.
    assert export.ignored_modules("mtp.layers.0.self_attn.q_proj.weight", (128, 128)) == (
        "mtp.layers.0.self_attn.qkv_proj",)
    # A conv is not a Linear the plugin is ever asked about.
    assert export.ignored_modules("model.visual.patch_embed.proj.weight", (64, 3, 2, 2, 2)) == ()
    assert export.ignored_modules("model.visual.blocks.0.norm1.weight", (64,)) == ()
    # A routed expert is ignored at the FusedMoE prefix, never at its leaves.
    assert export.ignored_modules(
        "model.language_model.layers.1.mlp.experts.7.gate_proj.weight", (64, 128)) == (
        "model.language_model.layers.1.mlp.experts",)


def test_the_exported_ignore_names_every_passed_through_linear(tmp_path, monkeypatch):
    """No encode (``--layers 0``): the question is the config, not the wire."""
    written = _export(tmp_path, monkeypatch, "--layers", "0")
    ignore = set(written["ignore"])
    missing = [m for m in VISION_MODULES if m not in ignore]
    assert not missing, (
        f"the exporter copies these Linears through at source precision and never names them, "
        f"so the plugin refuses the checkpoint at load: {missing}; ignore={sorted(ignore)}")
    leaves = [i for i in ignore if i.endswith(("mlp.gate_proj", "mlp.up_proj"))]
    assert not leaves, f"ignore names unmerged roles vLLM never builds as modules: {leaves}"
    assert "model.visual.patch_embed.proj" not in ignore, (
        "a Conv3d was named as a Linear")
    # The embedding is passed through like any other non-body tensor, so its
    # name comes from the tensor that was written, under whatever layout the
    # model has (#139): a hard-coded ``model.embed_tokens`` names a module
    # this nested checkpoint does not have and misses the one it does.
    assert "model.language_model.embed_tokens" in ignore, sorted(ignore)
    assert "model.embed_tokens" not in ignore, sorted(ignore)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the encoder is a GPU job")


@cuda
def test_the_vision_tower_is_named_beside_a_real_body_encode(tmp_path, monkeypatch):
    """The same, with the body actually encoded: declared and ignored partition.

    ``--passthrough-unrouted`` because this arm encodes the body, and the body
    contains ``o_proj``, which #99's construction gate refuses: the pinned
    runtime builds it with ``quant_config=None``, so a wire written there is
    dead weight.  The refusal is correct and stays default-on -- this test is
    about the ignore partition, not about the routing gate, so it takes the
    safe escape hatch (source precision) rather than suppressing the gate.
    The gate itself is tested in ``test_serving_export_gate.py``.
    """
    written = _export(tmp_path, monkeypatch, "--passthrough-unrouted")
    ignore = set(written["ignore"])
    declared = {t for g in written["config_groups"].values() for t in g["targets"]}
    assert declared, "nothing was quantized; this arm no longer tests the encode path"
    missing = [m for m in VISION_MODULES if m not in ignore]
    assert not missing, f"vision Linears neither declared nor ignored: {missing}"
    assert not (ignore & {t.rstrip("$") for t in declared}), (
        "a module is both declared and ignored; the plugin refuses that at parse")


def test_an_export_that_plans_nothing_is_refused(tmp_path, monkeypatch):
    """The same failure family: silent at export, refused at load.

    A menu no tensor here fits leaves ``plan`` empty, and the exporter used to
    write a checkpoint with an empty ``config_groups`` and report success.
    ``TesseraConfig.from_config`` refuses that ("a Tessera checkpoint declares
    its wires in config_groups"), hours later.  ``--layers 0`` stays legal: it
    is how a passthrough copy is asked for on purpose.
    """
    src = _write(tmp_path)
    out = tmp_path / "out"
    plan = tmp_path / "plan.json"
    # Every dense body weight passed through by name.
    plan.write_text(json.dumps({
        n: "PASSTHROUGH" for n in
        ("model.language_model.layers.0.self_attn.q_proj.weight",
         "model.language_model.layers.0.self_attn.k_proj.weight",
         "model.language_model.layers.0.self_attn.v_proj.weight",
         "model.language_model.layers.0.self_attn.o_proj.weight",
         "model.language_model.layers.0.mlp.gate_proj.weight",
         "model.language_model.layers.0.mlp.up_proj.weight",
         "model.language_model.layers.0.mlp.down_proj.weight")}))
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out), "--grid", "E4M3",
                                     "--q256", "1024", "--plan-json", str(plan)])
    with pytest.raises(SystemExit) as caught:
        export.main()
    assert "nothing was planned" in str(caught.value), str(caught.value)
    assert not out.exists() or not list(out.glob("*.safetensors")), (
        "the refusal fired only after writing the checkpoint")


def test_a_packed_expert_stack_is_named_in_either_spelling():
    """transformers-5 may store the stack as a parameter with no ``.weight``.

    The suffix is not what makes a rank-3 tensor at the experts path a stack,
    and ``module_of``-style string surgery on a name that never had the suffix
    chops seven characters off a real prefix.
    """
    packed = "model.language_model.layers.1.mlp.experts.gate_up_proj"
    assert export.ignored_modules(packed + ".weight", (4, 128, 128)) == (
        "model.language_model.layers.1.mlp.experts",)
    assert export.ignored_modules(packed, (4, 128, 128)) == (
        "model.language_model.layers.1.mlp.experts",)
    # A conv is rank 3 too, and sits nowhere near an experts path.
    assert export.ignored_modules(
        "model.language_model.layers.0.self_attn.k_conv1d.weight", (8192, 1, 4)) == ()
