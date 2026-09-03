"""What the exporter is allowed to call a dense Linear.

Three failures live here, and every one of them is silent at plan time and
expensive later:

1. **The body was not found at all.**  ``quantizable`` filtered on
   ``name.startswith("model.layers.")``.  A multimodal checkpoint roots its
   decoder under a sub-model -- GLM-5.3-Flash is
   ``model.language_model.layers.N.`` -- so the filter matched NOTHING, and an
   export of it would have reported success having quantized zero Linears.
2. **A routed expert planned as a dense Linear.**  Unpacked per-expert weights
   (``...mlp.experts.7.gate_proj.weight``, 864 per projection on this model)
   are 2-D, so nothing distinguished them from an ordinary Linear.  They would
   have been encoded -- hours of GPU -- into a checkpoint whose
   ``config_groups`` name modules vLLM never builds, which the plugin refuses
   at LOAD.  The refusal has to arrive before the encode, not after it.
3. **The ignore list named modules vLLM never builds.**  ``ignore`` is tested by
   EXACT membership against the prefix vLLM constructs a module at, so a name
   that is one character off is not a near miss -- it is a load-time refusal
   (or, for the MoE module, a refusal that no ignore entry can lift).  Two ways
   to get it wrong live here: naming a routed expert's 2592 leaves instead of
   the one FusedMoE prefix, and naming an unmerged ``gate_proj``/``up_proj``
   pair instead of the ``gate_up_proj`` vLLM merges them into.
4. **A conv1d called an expert stack.**  ``len(shape) >= 3`` was the whole
   test for "packed expert stack", and GLM-5.3-Flash's attention carries
   ``k_conv1d.weight [8192, 1, 4]``.  That put ``...self_attn`` -- the parent
   of every attention Linear in the layer -- into the checkpoint's ``ignore``
   list.

The packed 3-D layout has no source at hand, so ITS tests are synthetic and
say so; the unpacked tests are shaped exactly like the real checkpoint.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

_spec = importlib.util.spec_from_file_location(
    "export_tessera_serving",
    Path(__file__).resolve().parents[1] / "experiments" / "export_tessera_serving.py")
export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export)

HIDDEN, MOE_INTER, EXPERTS = 128, 64, 4
#: The dims that make GLM-5.3-Flash's packed orientation UNDECIDABLE, scaled
#: down: ``hidden == 2 * moe_intermediate``, so ``gate_up_proj`` is square.
AMBIGUOUS_HIDDEN, AMBIGUOUS_INTER = 128, 64


def _config(hidden=HIDDEN, inter=MOE_INTER):
    return {"architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {"hidden_size": hidden, "moe_intermediate_size": inter,
                            "num_hidden_layers": 2, "n_routed_experts": EXPERTS}}


def _write(tmp_path: Path, tensors: dict, config=None) -> Path:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    safetensors_torch.save_file({k: v.contiguous() for k, v in tensors.items()},
                                str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps(config or _config()))
    return src


def _unpacked_checkpoint():
    """Every tensor kind the real 4-layer GLM checkpoint has, in miniature."""
    t = {}
    P = "model.language_model.layers"
    # layer 0: a DENSE mlp, plus attention
    t[f"{P}.0.mlp.gate_proj.weight"] = torch.zeros(3 * HIDDEN, HIDDEN)
    t[f"{P}.0.mlp.up_proj.weight"] = torch.zeros(3 * HIDDEN, HIDDEN)
    t[f"{P}.0.mlp.down_proj.weight"] = torch.zeros(HIDDEN, 3 * HIDDEN)
    t[f"{P}.0.self_attn.o_proj.weight"] = torch.zeros(HIDDEN, HIDDEN)
    # ...and the conv1d that rank alone would call an expert stack
    t[f"{P}.0.self_attn.k_conv1d.weight"] = torch.zeros(2 * HIDDEN, 1, 4)
    # layer 1: a ROUTED MoE, with shared experts and a router beside it
    for e in range(EXPERTS):
        t[f"{P}.1.mlp.experts.{e}.gate_proj.weight"] = torch.zeros(MOE_INTER, HIDDEN)
        t[f"{P}.1.mlp.experts.{e}.up_proj.weight"] = torch.zeros(MOE_INTER, HIDDEN)
        t[f"{P}.1.mlp.experts.{e}.down_proj.weight"] = torch.zeros(HIDDEN, MOE_INTER)
    t[f"{P}.1.mlp.shared_experts.gate_proj.weight"] = torch.zeros(MOE_INTER, HIDDEN)
    t[f"{P}.1.mlp.shared_experts.up_proj.weight"] = torch.zeros(MOE_INTER, HIDDEN)
    t[f"{P}.1.mlp.shared_experts.down_proj.weight"] = torch.zeros(HIDDEN, MOE_INTER)
    t[f"{P}.1.mlp.gate.weight"] = torch.zeros(EXPERTS, HIDDEN)
    # the vision tower, which is not under ``layers.N.`` at all
    t["model.visual.blocks.0.mlp.gate_proj.weight"] = torch.zeros(HIDDEN, HIDDEN)
    t["model.visual.merger.proj.weight"] = torch.zeros(HIDDEN, HIDDEN)
    # and the two tensors that are never body Linears
    t["lm_head.weight"] = torch.zeros(HIDDEN, HIDDEN)
    t["model.embed_tokens.weight"] = torch.zeros(HIDDEN, HIDDEN)
    return t


def test_the_body_is_found_under_a_sub_model(tmp_path):
    """``model.language_model.layers.N.`` is a body; the old filter found none."""
    src = _write(tmp_path, _unpacked_checkpoint())
    _shards, shapes, packed, routed = export.quantizable(src)

    assert shapes, ("no dense Linear found under model.language_model.layers.*; the body prefix "
                    "test is back to startswith('model.layers.')")
    assert len(routed) == EXPERTS * 3, f"expected {EXPERTS * 3} routed expert leaves, got {len(routed)}"
    assert packed == {}, f"nothing here is a packed expert stack, but got {sorted(packed)}"


def test_a_routed_expert_is_never_a_dense_linear(tmp_path):
    """The split, at the leaf that separates routed from shared."""
    src = _write(tmp_path, _unpacked_checkpoint())
    _shards, shapes, _packed, routed = export.quantizable(src)

    routed_names = set(routed)
    assert all(".mlp.experts." in n for n in routed_names), sorted(routed_names)[:3]
    assert not any(".mlp.experts." in n for n in shapes), (
        "a routed expert reached the dense plan: "
        f"{[n for n in shapes if '.mlp.experts.' in n][:3]}")

    # ...and the shared experts beside them ARE dense Linears.  The ``\\d+``
    # segment in ROUTED_EXPERT_2D is the only thing separating the two.
    shared = [n for n in shapes if "shared_experts" in n]
    assert len(shared) == 3, f"the shared experts must stay quantizable as dense Linears: {shared}"


def test_a_conv1d_is_not_a_packed_expert_stack(tmp_path):
    """Rank 3 is not the test; the name is.

    Getting this wrong does not just skip a tensor: the module of
    ``...self_attn.k_conv1d.weight`` minus its leaf is ``...self_attn``, and
    that lands in the exported checkpoint's ``ignore``, covering every
    attention Linear in the layer.
    """
    src = _write(tmp_path, _unpacked_checkpoint())
    _shards, _shapes, packed, _routed = export.quantizable(src)

    assert packed == {}, (
        f"a conv1d was classified as a packed expert stack: {sorted(packed)}; its ignore entry "
        "would be the whole self_attn module")


def test_the_layer_index_survives_the_sub_model(tmp_path):
    """``name.split('.')[2]`` read 'layers' as an int on these names."""
    P = "model.language_model.layers"
    assert export.body_layer(f"{P}.0.mlp.gate_proj.weight") == 0
    assert export.body_layer(f"{P}.13.self_attn.o_proj.weight") == 13
    assert export.body_layer("model.layers.7.mlp.up_proj.weight") == 7, "a plain body still parses"


def test_planning_a_routed_expert_is_refused_before_any_encode(tmp_path, monkeypatch, capsys):
    """The refusal has to be reachable from the plan, not from the encode."""
    src = _write(tmp_path, _unpacked_checkpoint())
    out = tmp_path / "out"
    plan = tmp_path / "plan.json"
    routed = f"model.language_model.layers.1.mlp.experts.0.gate_proj.weight"
    plan.write_text(json.dumps({routed: {"grid": "E4M3", "q256": 1024}}))
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out), "--grid", "E4M3",
                                     "--q256", "1024", "--plan-json", str(plan)])

    with pytest.raises(SystemExit) as caught:
        export.main()

    message = str(caught.value)
    assert "ROUTED expert" in message, message
    assert "FusedMoE" in message, f"the refusal must say WHY vLLM cannot serve it: {message}"
    assert not out.exists() or not list(out.glob("*.safetensors")), (
        "the refusal fired only after writing weights")


# --------------------------------------------------------------------------
# The packed 3-D layout.  SYNTHETIC: no packed-expert source is at hand, so
# these fix the CONTRACT (which axis is the output, and when to refuse), not
# agreement with a real transformers-5 checkpoint.
# --------------------------------------------------------------------------

def test_packed_orientation_is_read_off_the_config_both_ways():
    config = _config(hidden=128, inter=48)                 # 2*48 = 96 != 128
    name = "model.language_model.layers.1.mlp.experts.gate_up_proj.weight"
    assert export.packed_expert_orientation(name, (EXPERTS, 96, 128), config) == "out_first"
    assert export.packed_expert_orientation(name, (EXPERTS, 128, 96), config) == "in_first"

    down = "model.language_model.layers.1.mlp.experts.down_proj.weight"
    assert export.packed_expert_orientation(down, (EXPERTS, 128, 48), config) == "out_first"
    assert export.packed_expert_orientation(down, (EXPERTS, 48, 128), config) == "in_first"


def test_packed_orientation_refuses_when_the_dims_cannot_decide():
    """The real case, not a contrived one.

    GLM-5.3-Flash has ``hidden_size == 2 * moe_intermediate_size``, so a
    packed ``gate_up_proj`` would be square and NO comparison of dims orients
    it.  A default axis order here transposes every expert in silence.
    """
    config = _config(hidden=AMBIGUOUS_HIDDEN, inter=AMBIGUOUS_INTER)
    name = "model.language_model.layers.1.mlp.experts.gate_up_proj.weight"
    with pytest.raises(SystemExit) as caught:
        export.packed_expert_orientation(name, (EXPERTS, 128, 128), config)
    assert "both axis orders fit" in str(caught.value), str(caught.value)


def test_packed_orientation_refuses_a_stack_that_fits_neither_way():
    config = _config(hidden=128, inter=48)
    name = "model.language_model.layers.1.mlp.experts.gate_up_proj.weight"
    with pytest.raises(SystemExit) as caught:
        export.packed_expert_orientation(name, (EXPERTS, 77, 55), config)
    assert "neither axis order fits" in str(caught.value), str(caught.value)


def test_a_packed_stack_is_recognised_by_name(tmp_path):
    """SYNTHETIC packed source: the split must route it to ``expert_shapes``."""
    t = _unpacked_checkpoint()
    for e in range(EXPERTS):                       # drop the unpacked leaves
        for proj in ("gate_proj", "up_proj", "down_proj"):
            t.pop(f"model.language_model.layers.1.mlp.experts.{e}.{proj}.weight")
    t["model.language_model.layers.1.mlp.experts.gate_up_proj.weight"] = torch.zeros(
        EXPERTS, 2 * MOE_INTER, HIDDEN)
    t["model.language_model.layers.1.mlp.experts.down_proj.weight"] = torch.zeros(
        EXPERTS, HIDDEN, MOE_INTER)
    src = _write(tmp_path, t)

    _shards, shapes, packed, routed = export.quantizable(src)
    assert sorted(packed) == [
        "model.language_model.layers.1.mlp.experts.down_proj.weight",
        "model.language_model.layers.1.mlp.experts.gate_up_proj.weight"], sorted(packed)
    assert routed == {}, sorted(routed)
    assert not any(".mlp.experts." in n for n in shapes), sorted(shapes)


# --------------------------------------------------------------------------
# ``ignore`` must name the modules vLLM BUILDS.
#
# The plugin's test is ``prefix in self.ignore`` -- exact membership against
# the string vLLM passes to ``get_quant_method``.  Both names below were read
# off the pinned build ``prismaquant/glm53-mia-sm121:487ecf187``:
#
#   models/glm5next/nvidia/model.py:239   FusedMoEFactory(prefix=f"{prefix}.experts")
#   models/glm5next/nvidia/model.py:124   Glm5NextMLP -> prefix=f"{prefix}.gate_up_proj"
#   models/glm5next/nvidia/model.py:216   shared_experts = Glm5NextMLP(prefix=f"{prefix}.shared_experts")
#   layers/fused_moe/layer.py:221         layer_name = prefix
#   layers/fused_moe/routed_experts.py:122,:201
#                                         quant_config.get_quant_method(self, self.layer_name)
# --------------------------------------------------------------------------

def test_a_shared_expert_fuses_its_gate_and_up_like_any_other_mlp():
    """``mlp.shared_experts`` is an MLP, and vLLM merges ITS gate/up too.

    The rule used to be scoped to ``.mlp.``, so these two leaves were declared
    as themselves -- two modules vLLM never builds -- while
    ``...shared_experts.gate_up_proj``, the one it does build, went undeclared
    and the plugin refused the load.
    """
    P = "model.language_model.layers.1.mlp"
    got = export.fused_module(f"{P}.shared_experts.gate_proj.weight")
    assert got is not None, "the shared expert's gate/up are not being fused"
    assert got[0] == f"{P}.shared_experts.gate_up_proj", got[0]
    assert got[1] == (f"{P}.shared_experts.gate_proj.weight",
                      f"{P}.shared_experts.up_proj.weight"), got[1]

    # ...and the ordinary dense case is unchanged.
    dense = export.fused_module("model.layers.0.mlp.up_proj.weight")
    assert dense[0] == "model.layers.0.mlp.gate_up_proj", dense[0]
    assert export.fused_module("model.layers.0.self_attn.k_proj.weight")[0] == \
        "model.layers.0.self_attn.qkv_proj"


def test_a_routed_expert_leaf_is_never_fused_as_a_dense_pair():
    """Their gate/up merge into ``w13`` INSIDE the FusedMoE, not into a Linear.

    Widening the fused rule off ``.mlp.`` would otherwise invent
    ``...mlp.experts.7.gate_up_proj``.
    """
    routed = "model.language_model.layers.1.mlp.experts.7.gate_proj.weight"
    assert export.fused_module(routed) is None, export.fused_module(routed)


@pytest.mark.parametrize("leaf", ["gate_proj", "up_proj", "down_proj"])
def test_the_routed_ignore_entry_is_the_fused_moe_prefix(leaf):
    """One entry per LAYER at ``...mlp.experts``, not one per checkpoint leaf."""
    name = f"model.language_model.layers.2.mlp.experts.13.{leaf}.weight"
    match = export.ROUTED_EXPERT_2D.match(name)
    assert match is not None, name
    assert match.group("moe") + ".experts" == "model.language_model.layers.2.mlp.experts"


#: These two go all the way through a real encode, which is a GPU job.  They
#: had no guard, so a host-safe run (``CUDA_VISIBLE_DEVICES=""``) reported them
#: RED for absent hardware -- a false red says nothing about the code and hides
#: a real one in the same colour.  Same spelling as test_merge_guard.py.
cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="the encoder is a GPU job")


@cuda
def test_the_exported_ignore_names_what_vllm_builds(tmp_path, monkeypatch):
    """End to end: run the exporter and read the ignore list it writes."""
    src = _write(tmp_path, _unpacked_checkpoint())
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out),
                                     "--grid", "E4M3", "--q256", "1024"])
    export.main()

    written = json.loads((out / "config.json").read_text())["quantization_config"]
    ignore = written["ignore"]
    declared = {t for g in written["config_groups"].values() for t in g["targets"]}

    P = "model.language_model.layers.1.mlp"
    assert f"{P}.experts" in ignore, (
        f"the FusedMoE prefix vLLM builds is absent from ignore, so the plugin refuses the "
        f"whole MoE layer at load: {sorted(ignore)}")
    leaves = [i for i in ignore if ".experts." in i]
    assert not leaves, f"ignore names routed expert LEAVES, which vLLM never asks about: {leaves[:3]}"

    # the shared experts are quantized, so they are DECLARED -- at gate_up_proj
    shared = sorted(t for t in declared if "shared_experts" in t)
    assert any("gate_up_proj" in t for t in shared), (
        f"the shared expert's merged module is not declared: {shared}")
    assert not any(t.rstrip("$").endswith("shared_experts.gate_proj")
                   or t.rstrip("$").endswith("shared_experts.up_proj") for t in declared), (
        "the shared expert's UNMERGED leaves are declared; vLLM builds no such module")


# --------------------------------------------------------------------------
# The MoE router.
#
# ``GateLinear(ReplicatedLinear)`` -- gate_linear.py:18 -- reads ``self.weight``
# directly at all six dispatch tiers (:179-228) and reads ``self.weight.dtype``
# to CHOOSE the tier (:84,:101,:128,:165,:174).  It never calls
# ``quant_method.apply``.  A Tessera method installed there is dead code and the
# ``weight`` it indexes is gone.
# --------------------------------------------------------------------------

def test_the_moe_router_is_not_a_projection():
    """``mlp.gate`` vs ``mlp.gate_proj`` is one underscore and two worlds."""
    assert export.MOE_ROUTER.match("model.language_model.layers.1.mlp.gate.weight")
    assert not export.MOE_ROUTER.match("model.language_model.layers.0.mlp.gate_proj.weight")
    assert not export.MOE_ROUTER.match(
        "model.language_model.layers.1.mlp.experts.3.gate_proj.weight")
    assert not export.MOE_ROUTER.match(
        "model.language_model.layers.1.mlp.shared_experts.gate_proj.weight")


def test_planning_the_router_is_refused_before_any_encode(tmp_path, monkeypatch):
    src = _write(tmp_path, _unpacked_checkpoint())
    out = tmp_path / "out"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "model.language_model.layers.1.mlp.gate.weight": {"grid": "E4M3", "q256": 1024}}))
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out), "--grid", "E4M3",
                                     "--q256", "1024", "--plan-json", str(plan)])
    with pytest.raises(SystemExit) as caught:
        export.main()
    message = str(caught.value)
    assert "ROUTER" in message, message
    assert "GateLinear" in message, f"the refusal must name the class that ignores the method: {message}"
    assert not out.exists() or not list(out.glob("*.safetensors"))


@cuda
def test_the_router_is_passed_through_and_ignored_by_default(tmp_path, monkeypatch):
    """The weight must survive into the checkpoint, unencoded.

    What matters is the passthrough, not the ignore entry: ``GateLinear`` takes
    no ``quant_config``, so this plugin is never asked about the router and the
    ignore entry is cosmetic (kept because it documents the passthrough and
    costs nothing).  What would break is the ENCODE -- it would take
    ``mlp.gate.weight`` out of the checkpoint and put wire tensors no loader
    maps in its place.

    The fixture's router is 4x128, which the shape rule would reject anyway, so
    this uses a router whose dims a default plan WOULD accept -- as the real
    model's 288x4096 one does.
    """
    tensors = _unpacked_checkpoint()
    tensors["model.language_model.layers.1.mlp.gate.weight"] = torch.zeros(64, HIDDEN)
    src = _write(tmp_path, tensors)
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out),
                                     "--grid", "E4M3", "--q256", "1024"])
    export.main()

    written = json.loads((out / "config.json").read_text())["quantization_config"]
    declared = {t for g in written["config_groups"].values() for t in g["targets"]}
    router = "model.language_model.layers.1.mlp.gate"
    assert not any(t.rstrip("$").endswith(".mlp.gate") for t in declared), (
        f"the router was declared as a Tessera target, so its weight is gone: {sorted(declared)}")
    with safetensors_torch.safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        assert f"{router}.weight" in set(handle.keys()), (
            "the router's weight is not in the exported checkpoint; the routing weight would be "
            "missing, not quantized")
    assert router in written["ignore"], (
        f"the router is not named in ignore: {sorted(written['ignore'])}")
