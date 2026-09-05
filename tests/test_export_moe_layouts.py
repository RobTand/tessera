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

The unpacked tests are shaped exactly like the real checkpoint.  The packed
3-D tests are synthetic in their CONFIGS -- they fix when an orientation is
decidable and when it must be refused, which needs dimension triples no one
checkpoint carries -- but the layout itself is not hypothetical: Qwen3.8-Flash-Next
on this box carries ``...mlp.experts.gate_up_proj [512, 1280, 2560]`` and
``...down_proj [512, 2560, 640]`` with no ``.weight`` suffix, and
``experiments/moe_plan_baseline.py`` classifies it as a real row beside
GLM-5.3-Flash-4layer's unpacked one.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

import box_artifacts

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


def _lfm_checkpoint():
    """The routed names in LiquidAI/LFM2.5-8B-A1B, in miniature."""
    generator = torch.Generator().manual_seed(19)

    def normal(*shape):
        return torch.randn(*shape, generator=generator) * 0.02

    stack = "model.layers.2.feed_forward.experts"
    tensors = {
        "model.layers.0.feed_forward.w1.weight": normal(2 * HIDDEN, HIDDEN),
        "model.layers.0.feed_forward.w3.weight": normal(2 * HIDDEN, HIDDEN),
        "model.layers.0.feed_forward.w2.weight": normal(HIDDEN, 2 * HIDDEN),
        "model.layers.2.feed_forward.gate.weight": normal(EXPERTS, HIDDEN),
        "model.layers.2.feed_forward.expert_bias": torch.zeros(EXPERTS),
        "lm_head.weight": normal(HIDDEN, HIDDEN),
        "model.embed_tokens.weight": normal(HIDDEN, HIDDEN),
    }
    for expert in range(EXPERTS):
        tensors[f"{stack}.{expert}.w1.weight"] = normal(MOE_INTER, HIDDEN)
        tensors[f"{stack}.{expert}.w3.weight"] = normal(MOE_INTER, HIDDEN)
        tensors[f"{stack}.{expert}.w2.weight"] = normal(HIDDEN, MOE_INTER)
    config = {
        "architectures": ["Lfm2MoeForCausalLM"],
        "hidden_size": HIDDEN,
        "intermediate_size": 2 * HIDDEN,
        "moe_intermediate_size": MOE_INTER,
        "num_hidden_layers": 3,
        "num_dense_layers": 2,
        "num_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "use_expert_bias": True,
    }
    return tensors, config


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


def test_lfm_expert_spellings_form_one_canonical_runtime_stack(tmp_path):
    """LFM names the same runtime shards ``w1``/``w3``/``w2`` on disk.

    The exporter must preserve those checkpoint spellings for vLLM's
    per-parameter mapping while declaring the canonical gate/up/down roles the
    Tessera container and routed-MoE scheme read.
    """
    tensors, config = _lfm_checkpoint()
    src = _write(tmp_path, tensors, config)
    _shards, shapes, packed, routed = export.quantizable(src)
    stack = "model.layers.2.feed_forward.experts"

    assert packed == {}
    assert len(routed) == EXPERTS * 3, sorted(routed)
    assert not [name for name in shapes if name.startswith(f"{stack}.")], (
        "LFM experts reached the dense Linear plan")
    stacks = export.expert_stacks(routed)
    assert sorted(stacks) == [stack]
    assert sorted(stacks[stack][0]) == sorted(export.EXPERT_PROJECTIONS)
    assert stacks[stack][0]["gate_proj"][0].endswith(".0.w1.weight")
    assert stacks[stack][0]["up_proj"][0].endswith(".0.w3.weight")
    assert stacks[stack][0]["down_proj"][0].endswith(".0.w2.weight")

    planned = export.plan_expert_stack(
        stack, stacks[stack], export.grid_for("E4M3"), 1024
    )
    by_role = {unit["projection"]: unit for unit in planned["units"]
               if unit["expert"] == 0}
    assert by_role["gate_proj"]["wire"].endswith(".0.w1.wire")
    assert by_role["up_proj"]["wire"].endswith(".0.w3.wire")
    assert by_role["down_proj"]["wire"].endswith(".0.w2.wire")
    assert export.MOE_ROUTER.match("model.layers.2.feed_forward.gate.weight")


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
# The packed 3-D layout.  The orientation edge cases below are synthetic so
# both axis orders and the square ambiguity can be exercised; the real
# transformers-5 Qwen source farther down pins the on-disk spelling and
# population separately.
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


#: Decidable dims the E4M3 encoder can also cut: ``2 * 32 != 128``, so a
#: packed ``gate_up_proj`` orients, and both groups' rows are whole tuples
#: (``grid.arity * 32 == 32``) with columns a multiple of 16.  ``PACKED_INTER``
#: below orients too but is 48, which the encoder cannot cut -- fine for the
#: classification tests that use it, useless for one that must reach the plan.
DECIDABLE_HIDDEN, DECIDABLE_INTER = 128, 32

#: The two closed conventions, as the shapes each one's ``gate_up_proj`` and
#: ``down_proj`` must have, and the orientation both of them state.
_LAYOUT_SHAPES = {
    "out_first_chunked": ("out_first", lambda h, i: {"gate_up_proj": (EXPERTS, 2 * i, h),
                                                     "down_proj": (EXPERTS, h, i)}),
    "in_first_interleaved": ("in_first", lambda h, i: {"gate_up_proj": (EXPERTS, h, 2 * i),
                                                       "down_proj": (EXPERTS, i, h)}),
}


def _packed_sources(shapes: dict):
    """``packed_expert_stacks`` over one stack's physical projections."""
    stack = "model.language_model.layers.1.mlp.experts"
    named = {f"{stack}.{projection}.weight": shape for projection, shape in shapes.items()}
    return stack, export.packed_expert_stacks(named)[stack]


@pytest.mark.parametrize("source_layout", sorted(_LAYOUT_SHAPES))
def test_the_declared_layout_and_the_inferred_orientation_are_one_predicate(source_layout):
    """The plan's shape table IS ``packed_expert_orientation``, not a second opinion.

    #136 asks whether the exporter answers "which axis is the output" twice:
    once by inference from the config dims, once by the plan's declaration.
    It does not.  ``plan_packed_expert_stack`` accepts EXACTLY the shape
    triple the declared convention implies from ``hidden_size`` and
    ``moe_intermediate_size``, and ``packed_expert_orientation`` returns
    "out_first" on precisely ``a == out and b == in`` -- the same comparison
    over the same two numbers, for both physical projections and both
    conventions.  So a decidable contradiction is already refused, by the
    shape check, before any encode; the inference cannot disagree with the
    declaration because it is the same question asked in the same terms.

    This is the check rather than a call from the plan into the inference:
    such a call would add no refusal (the case it would catch is refused
    already) and, on the ambiguous shapes below, would refuse the one case an
    explicit declaration exists for.  A guard that can only fire where
    something else already refused, and misfires where it does not, is not a
    gate.
    """
    hidden, inter = DECIDABLE_HIDDEN, DECIDABLE_INTER
    config = _config(hidden=hidden, inter=inter)
    orientation, declared_shapes = _LAYOUT_SHAPES[source_layout]
    other_layout = next(name for name in _LAYOUT_SHAPES if name != source_layout)
    other_orientation, contradicting_shapes = _LAYOUT_SHAPES[other_layout]

    # The inference reads the declared convention's shapes as its orientation,
    # and the other convention's as the opposite one. Both are decidable here.
    stack, sources = _packed_sources(declared_shapes(hidden, inter))
    for projection, (name, shape) in sources.items():
        assert export.packed_expert_orientation(name, shape, config) == orientation
    _stack, contradicting = _packed_sources(contradicting_shapes(hidden, inter))
    for projection, (name, shape) in contradicting.items():
        assert export.packed_expert_orientation(name, shape, config) == other_orientation

    # The plan accepts the first and refuses the second, naming the declared
    # convention and the shape it implies -- the refusal the inference would
    # have produced, in the planner's own words.
    planned = export.plan_packed_expert_stack(
        stack, sources, export.grid_for("E4M3"), 1024,
        source_layout=source_layout, config=config)
    assert planned["source_layout"] == source_layout
    assert len(planned["units"]) == EXPERTS * 3

    with pytest.raises(SystemExit) as caught:
        export.plan_packed_expert_stack(
            stack, contradicting, export.grid_for("E4M3"), 1024,
            source_layout=source_layout, config=config)
    message = str(caught.value)
    assert f"source_layout={source_layout!r}" in message, message
    assert "from config.json" in message, message


@pytest.mark.parametrize("source_layout", sorted(_LAYOUT_SHAPES))
def test_an_explicit_layout_survives_an_ambiguous_shape(source_layout):
    """Where the dims cannot decide, the declaration must -- and does.

    ``hidden == 2 * moe_intermediate`` is GLM-5.3-Flash's own geometry, and it
    makes ``gate_up_proj`` square: the two conventions' shape triples coincide
    and the inference refuses (``both axis orders fit``).  The plan does not,
    because the plan is not inferring.  This is the case that makes calling
    the inference from the plan path wrong rather than redundant.
    """
    hidden, inter = AMBIGUOUS_HIDDEN, AMBIGUOUS_INTER
    config = _config(hidden=hidden, inter=inter)
    _orientation, declared_shapes = _LAYOUT_SHAPES[source_layout]
    stack, sources = _packed_sources(declared_shapes(hidden, inter))
    assert sources["gate_up_proj"][1] == (EXPERTS, hidden, hidden), sources["gate_up_proj"]

    with pytest.raises(SystemExit) as caught:
        export.packed_expert_orientation(*sources["gate_up_proj"], config)
    assert "both axis orders fit" in str(caught.value), str(caught.value)

    planned = export.plan_packed_expert_stack(
        stack, sources, export.grid_for("E4M3"), 1024,
        source_layout=source_layout, config=config)
    assert planned["source_layout"] == source_layout
    assert len(planned["units"]) == EXPERTS * 3


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


@pytest.mark.parametrize(
    "source_layout,suffix",
    [("out_first_chunked", ".weight"), ("in_first_interleaved", "")],
)
def test_explicit_packed_layouts_slice_to_canonical_expert_matrices(
        source_layout, suffix):
    """Both packed conventions become the same gate/up/down matrices.

    The tensor dimensions alone are deliberately ambiguous here
    (``hidden == 2 * intermediate``).  The plan's explicit convention—not a
    dimension heuristic—must decide the axis order and gate/up split.
    """
    stack = "model.language_model.layers.1.mlp.experts"
    gate = torch.arange(EXPERTS * MOE_INTER * HIDDEN, dtype=torch.float32).reshape(
        EXPERTS, MOE_INTER, HIDDEN)
    up = gate + gate.numel()
    down = (torch.arange(EXPERTS * HIDDEN * MOE_INTER, dtype=torch.float32)
            .reshape(EXPERTS, HIDDEN, MOE_INTER) + 2 * gate.numel())
    if source_layout == "out_first_chunked":
        gate_up_source = torch.cat((gate, up), dim=1)
        down_source = down
    else:
        gate_up_source = torch.empty(EXPERTS, HIDDEN, 2 * MOE_INTER)
        gate_up_source[:, :, 0::2] = gate.transpose(1, 2)
        gate_up_source[:, :, 1::2] = up.transpose(1, 2)
        down_source = down.transpose(1, 2).contiguous()
    sources = {
        f"{stack}.gate_up_proj{suffix}": gate_up_source,
        f"{stack}.down_proj{suffix}": down_source,
    }
    packed = export.packed_expert_stacks(
        {name: tuple(tensor.shape) for name, tensor in sources.items()})
    planned = export.plan_packed_expert_stack(
        stack, packed[stack], export.grid_for("E4M3"), 1024,
        source_layout=source_layout, config=_config())

    expected = {"gate_proj": gate, "up_proj": up, "down_proj": down}
    assert planned["source_layout"] == source_layout
    assert len(planned["units"]) == EXPERTS * 3
    for unit in planned["units"]:
        actual = export.packed_expert_weight(sources[unit["source_tensor"]], unit)
        assert torch.equal(actual, expected[unit["projection"]][unit["expert"]])
        assert unit["tensor"] == f"{stack}.{unit['expert']}.{unit['projection']}.weight"
        assert unit["wire"] == f"{stack}.{unit['expert']}.{unit['projection']}.wire"
        assert unit["source_layout"] == source_layout
        assert unit["source_slice"]["expert"] == unit["expert"]


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


@pytest.mark.parametrize("role", ["w1", "w3"])
def test_lfm_dense_gate_up_names_the_constructed_w13(role):
    prefix = "model.layers.0.feed_forward"
    name = f"{prefix}.{role}.weight"
    assert export.fused_module(name) == (
        f"{prefix}.w13", (f"{prefix}.w1.weight", f"{prefix}.w3.weight"))
    assert export.ignored_modules(name, (64, 128)) == (f"{prefix}.w13",)
    routed = f"model.layers.2.feed_forward.experts.7.{role}.weight"
    assert export.fused_module(routed) is None
    assert export.ignored_modules(routed, (64, 128)) == (
        "model.layers.2.feed_forward.experts",)


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
    # --passthrough-unrouted: the fixture's body carries ``o_proj``, which #99's
    # construction gate refuses because the pinned runtime builds it with
    # ``quant_config=None``.  The refusal is correct and stays default-on; this
    # test is about the MoE ignore/passthrough layout, not the routing gate, so
    # it takes the safe escape hatch rather than suppressing the gate.
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out),
                                     "--grid", "E4M3", "--q256", "1024",
                                     "--passthrough-unrouted"])
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
    # --passthrough-unrouted: the fixture's body carries ``o_proj``, which #99's
    # construction gate refuses because the pinned runtime builds it with
    # ``quant_config=None``.  The refusal is correct and stays default-on; this
    # test is about the MoE ignore/passthrough layout, not the routing gate, so
    # it takes the safe escape hatch rather than suppressing the gate.
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out),
                                     "--grid", "E4M3", "--q256", "1024",
                                     "--passthrough-unrouted"])
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


# --------------------------------------------------------------------------
# The packed stack that carries NO ``.weight`` suffix.
#
# transformers-5 stores a layer's packed experts as an ``nn.Parameter`` on the
# experts module, not as a child Linear's ``weight``, so the tensor on disk is
# ``...mlp.experts.gate_up_proj`` -- 96 body tensors of exactly that spelling
# on ``Qwen3.8-Flash-Next`` under the shared models root
# (``TESSERA_SHARED_MODELS_DIR``), the one packed source on this box.  ``quantizable`` tested ``name.endswith(".weight")`` before it looked at
# anything else, so those tensors were classified as NOTHING: not dense, not
# packed, not routed.  Being absent from ``expert_shapes`` is what makes it
# expensive -- ``ignore`` is built from that dict, so the FusedMoE module was
# never named, and ``TesseraConfig.get_quant_method`` refuses a routed-MoE
# layer it does not find in ``ignore``.  The export therefore succeeds, the
# dense body encodes for hours, and the checkpoint refuses at load.
#
# ``packed_expert_orientation`` already read both spellings
# (``name.endswith("gate_up_proj.weight") or name.endswith("gate_up_proj")``),
# which is how far the inconsistency reached before anything caught it.
# --------------------------------------------------------------------------

#: An intermediate size that ORIENTS: ``2 * 48 != 128``, so a packed
#: ``gate_up_proj`` is not square and ``packed_expert_orientation`` decides it
#: instead of refusing.  These tests are about the plan-time classification, so
#: the orientation must not be the thing that raises.
PACKED_INTER = 48


def _packed_checkpoint(suffix: str):
    """The unpacked fixture with layer 1's experts replaced by a packed stack.

    ``suffix`` is ``".weight"`` or ``""`` -- the two spellings a packed stack
    is written under.  Everything else is identical, so a test that runs both
    is comparing the spelling and nothing else.
    """
    t = _unpacked_checkpoint()
    for e in range(EXPERTS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            t.pop(f"model.language_model.layers.1.mlp.experts.{e}.{proj}.weight")
    t[f"model.language_model.layers.1.mlp.experts.gate_up_proj{suffix}"] = torch.zeros(
        EXPERTS, 2 * PACKED_INTER, HIDDEN)
    t[f"model.language_model.layers.1.mlp.experts.down_proj{suffix}"] = torch.zeros(
        EXPERTS, HIDDEN, PACKED_INTER)
    return t


@pytest.mark.parametrize("suffix", [".weight", ""])
def test_a_packed_stack_is_found_under_either_spelling(tmp_path, suffix):
    """Both spellings land in ``expert_shapes``; neither reaches ``shapes``."""
    src = _write(tmp_path, _packed_checkpoint(suffix), _config(inter=PACKED_INTER))

    _shards, shapes, packed, routed = export.quantizable(src)

    assert sorted(packed) == [
        f"model.language_model.layers.1.mlp.experts.down_proj{suffix}",
        f"model.language_model.layers.1.mlp.experts.gate_up_proj{suffix}"], sorted(packed)
    assert routed == {}, sorted(routed)
    assert not any(".mlp.experts." in n for n in shapes), sorted(shapes)


@pytest.mark.parametrize("suffix", [".weight", ""])
def test_planning_a_packed_stack_is_refused_before_any_encode(tmp_path, monkeypatch, suffix):
    """A plan naming a packed stack refuses at PLAN time, under either spelling.

    Without the suffix the stack was in no bucket at all, so the plan-time
    guard could not see it either: the name fell through to the ``unknown``
    check and refused there, naming no MoE reason -- or, on a checkpoint that
    also had a matching dense name, would not have refused at all.
    """
    src = _write(tmp_path, _packed_checkpoint(suffix), _config(inter=PACKED_INTER))
    out = tmp_path / "out"
    plan = tmp_path / "plan.json"
    stack = f"model.language_model.layers.1.mlp.experts.gate_up_proj{suffix}"
    plan.write_text(json.dumps({stack: {"grid": "E4M3", "q256": 1024}}))
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out), "--grid", "E4M3",
                                     "--q256", "1024", "--plan-json", str(plan)])

    with pytest.raises(SystemExit) as caught:
        export.main()

    message = str(caught.value)
    assert "packed expert tensor" in message, message
    assert stack in message, message
    assert not out.exists() or not list(out.glob("*.safetensors")), (
        "the refusal fired only after writing weights")


@pytest.mark.parametrize("source_layout", [None, "guessed_from_shape"])
def test_a_packed_stack_plan_requires_a_supported_source_layout(
        tmp_path, monkeypatch, source_layout):
    """Missing and invented packed conventions both fail before encoding."""
    src = _write(tmp_path, _packed_checkpoint(""), _config(inter=PACKED_INTER))
    out = tmp_path / "out"
    plan = tmp_path / "plan.json"
    stack = "model.language_model.layers.1.mlp.experts"
    spec = {"grid": "E4M3", "q256": 1024}
    if source_layout is not None:
        spec["source_layout"] = source_layout
    plan.write_text(json.dumps({stack: spec}))
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out), "--grid", "E4M3",
                                     "--q256", "1024", "--plan-json", str(plan)])

    with pytest.raises(SystemExit, match=(
            "source_layout.*out_first_chunked.*in_first_interleaved")):
        export.main()
    assert not out.exists() or not list(out.glob("*.safetensors"))


#: The one packed-expert source on this box.  Skipped rather than synthesised
#: where it is absent: the point of this test is that the spelling on REAL
#: disk is the one the classifier missed, and a fixture cannot say that.  The
#: root is ``box_artifacts``' to name, so the skip reason names the variable
#: that moves it and ``--strict-cuda`` can refuse a box that lacks it.
QWEN_PACKED = box_artifacts.path("shared_models", "Qwen3.8-Flash-Next")


@box_artifacts.require("shared_models", "Qwen3.8-Flash-Next",
                       "model.safetensors.index.json")
def test_the_real_packed_source_is_classified_as_experts():
    """96 body packed stacks, none of them planned as a dense Linear.

    48 decoder layers x {gate_up_proj, down_proj}, every one of them written
    with no ``.weight``.  The two remaining stacks in the checkpoint are the
    MTP sidecar's (``mtp.layers.0.mlp.experts.*``), which ``BODY_LAYER`` does
    not match by the same rule that keeps the vision tower out.
    """
    _shards, shapes, packed, routed = export.quantizable(QWEN_PACKED)

    assert len(packed) == 96, sorted(packed)[:4]
    assert all(n.endswith(("gate_up_proj", "down_proj")) for n in packed), sorted(packed)[:4]
    assert not any(".mlp.experts." in n or n.endswith(".mlp.experts") for n in shapes), \
        [n for n in shapes if ".experts" in n][:4]
    assert routed == {}, sorted(routed)[:4]
    # the ignore names come off ``ignored_modules`` (#86), the one rule, which
    # reads a bare packed name through its ``.weight`` spelling
    modules = {m for n, shape in packed.items() for m in export.ignored_modules(n, shape)}
    assert len(modules) == 48, sorted(modules)[:4]
    assert all(m.endswith(".mlp.experts") for m in modules), sorted(modules)[:4]


def test_a_rank_2_bare_packed_name_is_refused_not_guessed(tmp_path):
    """Admitting the bare spelling widened the classifier; this pins the edge.

    ``quantizable`` now reads ``<moe>.experts.<projection>`` with no
    ``.weight`` (through the ``.weight``-spelled probe), and the only
    checkpoint layout known to write that spelling writes a 3-D stack.  A 2-D tensor under the same name is a layout this
    exporter has not been shown, and both available answers are wrong in the
    expensive direction: filed as an expert stack it leaves the module BF16
    and named in ``ignore``; filed as a dense Linear it puts a module in
    ``config_groups`` that vLLM never builds, which the plugin refuses at
    LOAD -- after the encode.  So it refuses, at plan time, by name and rank.

    This is the same failure the ``len(shape) >= 3`` rule already made once
    (a ``k_conv1d.weight`` called an expert stack, docstring item 4); the
    lesson is not to trade one silent rank assumption for another.
    """
    t = _unpacked_checkpoint()
    for e in range(EXPERTS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            t.pop(f"model.language_model.layers.1.mlp.experts.{e}.{proj}.weight")
    bare = "model.language_model.layers.1.mlp.experts.gate_up_proj"
    t[bare] = torch.zeros(2 * PACKED_INTER, HIDDEN)     # rank 2, bare name
    src = _write(tmp_path, t, _config(inter=PACKED_INTER))

    with pytest.raises(SystemExit) as caught:
        export.quantizable(src)

    message = str(caught.value)
    assert bare in message, message
    assert "rank 2" in message, message


@pytest.mark.parametrize("suffix", [".weight", ""])
def test_a_packed_stack_is_found_under_a_feed_forward_owner(tmp_path, suffix):
    """The owner is ``mlp`` on GLM and ``feed_forward`` on LFM -- the file says
    so beside ``ROUTED_EXPERT_2D`` and ``MOE_ROUTER``, both of which accept
    either.  ``PACKED_EXPERT_ND`` accepted only ``mlp``, so a transformers-5
    packed stack under ``feed_forward`` landed in no bucket at all and could not
    be refused by name at plan time."""
    tensors = {name.replace(".mlp.experts.", ".feed_forward.experts."): value
               for name, value in _packed_checkpoint(suffix).items()}
    src = _write(tmp_path, tensors, _config(inter=PACKED_INTER))

    _shards, shapes, packed, routed = export.quantizable(src)

    assert sorted(packed) == [
        f"model.language_model.layers.1.feed_forward.experts.down_proj{suffix}",
        f"model.language_model.layers.1.feed_forward.experts.gate_up_proj{suffix}"], sorted(packed)
    assert routed == {}, sorted(routed)
    assert not any(".feed_forward.experts." in n for n in shapes), sorted(shapes)
