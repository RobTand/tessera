"""The exporter's routed-MoE write half: what it writes, and what it refuses first.

Until 2026-09-04 no Tessera checkpoint could contain a routed-MoE expert
stack.  The plugin's expert route existed and had been loaded and executed on
the pinned runtime; the sidecar shape existed; the parameter layout existed --
and the exporter wrote none of it, so nothing could be served through any of
them.  This file covers the half that was missing and the seam it closes: what
the exporter WRITES is exactly what ``tessera.serving`` READS, checked with the
serving code itself rather than with a second copy of its rules.

Two properties are worth stating out loud, because getting either wrong is
silent:

* **The stack is the plannable unit.**  vLLM builds ONE method per
  ``RoutedExperts`` module, so a plan entry names ``<moe>.experts`` and gives
  every expert of that stack one rung.  A plan naming a single expert's
  ``gate_proj`` describes half a module the runtime builds whole, and is
  refused with the spelling that works.
* **Not planning a stack changes nothing.**  An unplanned stack stays at
  source precision and is named in ``ignore`` at the FusedMoE prefix, exactly
  as before, so every checkpoint written before this is byte-identical under
  the same command line.  That control is a test here, not an assurance.

The refusals are all PLAN-TIME.  A routed stack on GLM-5.3-Flash is 864 units
and roughly 75 minutes of GPU per layer; a refusal that arrives after the
encode is not a refusal, it is a bill.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from tessera.moe_layout import MoePacked, unpack_moe_wires  # noqa: E402
from tessera.serving.contract import (classify_construction,  # noqa: E402
                                      construction_entry)
from tessera.serving.scheme import (MOE_GROUPS, expert_role_declarations,  # noqa: E402
                                    parse_tessera_expert_blob,
                                    validate_tessera_moe_scheme)

_spec = importlib.util.spec_from_file_location(
    "export_tessera_serving",
    Path(__file__).resolve().parents[1] / "experiments" / "export_tessera_serving.py")
export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export)

HIDDEN, MOE_INTER, EXPERTS = 128, 64, 4
LAYER = "model.language_model.layers.1"
STACK = f"{LAYER}.mlp.experts"

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="the encoder is a GPU job")


def _config():
    return {"architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {"hidden_size": HIDDEN, "moe_intermediate_size": MOE_INTER,
                            "num_hidden_layers": 2, "n_routed_experts": EXPERTS}}


def _checkpoint(experts=EXPERTS, projections=("gate_proj", "up_proj", "down_proj"),
                skip=(), inter=MOE_INTER):
    """A miniature of the real 4-layer GLM: one dense layer, one routed MoE layer.

    The weights are RANDOM rather than zero: a trellis over an all-zero tile
    encodes to a degenerate scale plane, and a round trip that only ever sees
    zeros proves nothing about a wire.
    """
    generator = torch.Generator().manual_seed(11)

    def normal(*shape):
        return torch.randn(*shape, generator=generator) * 0.02

    tensors = {
        f"model.language_model.layers.0.mlp.gate_proj.weight": normal(2 * HIDDEN, HIDDEN),
        f"model.language_model.layers.0.mlp.up_proj.weight": normal(2 * HIDDEN, HIDDEN),
        f"model.language_model.layers.0.mlp.down_proj.weight": normal(HIDDEN, 2 * HIDDEN),
        f"{LAYER}.mlp.shared_experts.gate_proj.weight": normal(inter, HIDDEN),
        f"{LAYER}.mlp.shared_experts.up_proj.weight": normal(inter, HIDDEN),
        f"{LAYER}.mlp.shared_experts.down_proj.weight": normal(HIDDEN, inter),
        f"{LAYER}.mlp.gate.weight": normal(experts, HIDDEN),
        "lm_head.weight": normal(HIDDEN, HIDDEN),
        "model.embed_tokens.weight": normal(HIDDEN, HIDDEN),
    }
    for index in range(experts):
        for projection in projections:
            if (index, projection) in skip:
                continue
            shape = (HIDDEN, inter) if projection == "down_proj" else (inter, HIDDEN)
            tensors[f"{STACK}.{index}.{projection}.weight"] = normal(*shape)
    return tensors


def _write(tmp_path: Path, tensors, config=None) -> Path:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    safetensors_torch.save_file({k: v.contiguous() for k, v in tensors.items()},
                                str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps(config or _config()))
    return src


def _export(tmp_path, monkeypatch, tensors, plan, *extra):
    """Run the exporter over ``tensors`` with ``plan``; return the out dir."""
    src = _write(tmp_path, tensors)
    out = tmp_path / "out"
    argv = ["export", str(src), str(out), "--grid", "E4M3", "--q256", "1024",
            # The synthetic architecture is censused, and its attention
            # projections are ``never_offered`` on the pinned runtime.  Passing
            # them through at source precision is the SAFE resolution and the
            # one this fixture wants; it is not a way around the gate, which
            # still runs and still names them.
            "--passthrough-unrouted", *extra]
    if plan is not None:
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan))
        argv += ["--plan-json", str(plan_path)]
    monkeypatch.setattr("sys.argv", argv)
    export.main()
    return out


# --------------------------------------------------------------------------
# The stack is the plannable unit
# --------------------------------------------------------------------------

def test_the_leaves_of_one_layer_group_into_one_stack(tmp_path):
    _shards, _shapes, _packed, routed = export.quantizable(_write(tmp_path, _checkpoint()))
    stacks = export.expert_stacks(routed)

    assert sorted(stacks) == [STACK], sorted(stacks)
    assert sorted(stacks[STACK]) == list(range(EXPERTS))
    assert sorted(stacks[STACK][0]) == sorted(export.EXPERT_PROJECTIONS)


def test_the_group_row_order_comes_off_the_runtimes_own_table():
    """``w13`` is gate then up because ``MOE_GROUP_SHARDS`` says ``w1`` then ``w3``."""
    assert export.MOE_GROUP_PROJECTIONS["w13"] == ("gate_proj", "up_proj")
    assert export.MOE_GROUP_PROJECTIONS["w2"] == ("down_proj",)
    assert export.EXPERT_PROJECTIONS == ("gate_proj", "up_proj", "down_proj")


def test_planning_a_leaf_is_refused_and_names_the_stack_spelling(tmp_path, monkeypatch):
    """The old refusal said the write half did not exist. It does; the plan is misspelled."""
    with pytest.raises(SystemExit) as caught:
        _export(tmp_path, monkeypatch, _checkpoint(),
                {f"{STACK}.0.gate_proj.weight": {"grid": "E4M3", "q256": 1024}})

    message = str(caught.value)
    assert "ROUTED expert" in message, message
    assert STACK in message, f"the refusal must name the spelling that works: {message}"


# --------------------------------------------------------------------------
# What is refused before the first encode
# --------------------------------------------------------------------------

def test_a_family_with_no_expert_route_is_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as caught:
        _export(tmp_path, monkeypatch, _checkpoint(),
                {STACK: {"grid": "E2M1x2", "q256": 896}})

    message = str(caught.value)
    assert "MOE_BUILDERS" in message and "TESSERA_FP8" in message, message


def test_a_gap_in_the_expert_indices_is_refused(tmp_path, monkeypatch):
    tensors = _checkpoint()
    for projection in export.EXPERT_PROJECTIONS:
        del tensors[f"{STACK}.1.{projection}.weight"]

    with pytest.raises(SystemExit) as caught:
        _export(tmp_path, monkeypatch, tensors, {STACK: {"grid": "E4M3", "q256": 1024}})

    assert "missing expert" in str(caught.value), str(caught.value)


def test_a_missing_projection_is_refused(tmp_path, monkeypatch):
    tensors = _checkpoint(skip=((2, "up_proj"),))

    with pytest.raises(SystemExit) as caught:
        _export(tmp_path, monkeypatch, tensors, {STACK: {"grid": "E4M3", "q256": 1024}})

    message = str(caught.value)
    assert "up_proj" in message and "gate/up" in message, message


def test_experts_that_disagree_about_geometry_are_refused(tmp_path, monkeypatch):
    tensors = _checkpoint()
    tensors[f"{STACK}.3.gate_proj.weight"] = torch.zeros(MOE_INTER * 2, HIDDEN)

    with pytest.raises(SystemExit) as caught:
        _export(tmp_path, monkeypatch, tensors, {STACK: {"grid": "E4M3", "q256": 1024}})

    assert "one tile" in str(caught.value), str(caught.value)


def test_a_stack_the_encoder_cannot_cut_is_refused_not_half_passed_through(tmp_path, monkeypatch):
    """A dense Linear that fails this is passed through; a stack cannot be."""
    tensors = _checkpoint(inter=40)                     # 40 % 32 != 0

    with pytest.raises(SystemExit) as caught:
        _export(tmp_path, monkeypatch, tensors, {STACK: {"grid": "E4M3", "q256": 1024}})

    message = str(caught.value)
    assert "ONE method per stack" in message, message


def test_a_planned_stack_outside_the_layer_bound_is_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as caught:
        _export(tmp_path, monkeypatch, _checkpoint(),
                {STACK: {"grid": "E4M3", "q256": 1024}}, "--layers", "1")

    assert "--layers 1" in str(caught.value), str(caught.value)


def test_a_stack_and_a_stock_twin_together_are_refused(tmp_path, monkeypatch):
    """The twin has no per-channel FP8 expert writer, so it would silently lose them."""
    with pytest.raises(SystemExit) as caught:
        _export(tmp_path, monkeypatch, _checkpoint(), {STACK: {"grid": "E4M3", "q256": 1024}},
                "--stock-twin", str(tmp_path / "twin"))

    assert "--stock-twin" in str(caught.value), str(caught.value)


# --------------------------------------------------------------------------
# The construction gate, on the stack
# --------------------------------------------------------------------------

def test_the_runtime_offers_the_expert_stack_a_quant_config():
    """Read off the census receipt's own table, not asserted here (principle 14).

    ``offered`` holds the ``LinearBase`` rows only; a ``RoutedExperts`` stack
    is recorded in ``offered_non_linear``, and a classifier that read only the
    first list called the one module the expert route exists for ``absent`` --
    refusing it while the same receipt said the runtime asks about it.
    """
    entry = construction_entry(["Glm5NextForConditionalGeneration"])
    assert entry is not None, "the pinned GLM census receipt is not in the contract"
    patterns = {row["prefix_pattern"] for row in entry["offered_non_linear"]}
    assert "language_model.model.layers.*.mlp.experts" in patterns, sorted(patterns)

    verdict, pattern = classify_construction(entry, STACK)
    assert verdict == "offered", (verdict, pattern)
    assert pattern == "language_model.model.layers.*.mlp.experts", pattern


# --------------------------------------------------------------------------
# The write half, end to end
# --------------------------------------------------------------------------

@cuda
def test_an_unplanned_stack_is_still_passed_through_and_ignored(tmp_path, monkeypatch):
    """The control: no plan entry, and nothing about the stack moves."""
    out = _export(tmp_path, monkeypatch, _checkpoint(), None)

    written = json.loads((out / "config.json").read_text())["quantization_config"]
    assert STACK in written["ignore"], sorted(written["ignore"])
    declared = {t for g in written["config_groups"].values() for t in g["targets"]}
    assert STACK not in declared
    with safetensors_torch.safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        names = set(handle.keys())
    assert not [n for n in names if n.endswith(".wire")], sorted(n for n in names if ".wire" in n)
    assert f"{STACK}.0.gate_proj.weight" in names, "the source expert weight was dropped"

    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert manifest["routed_moe"]["disposition"] == "passed_through_bf16"


@cuda
def test_a_planned_stack_is_written_as_the_plugin_reads_it(tmp_path, monkeypatch):
    """The seam: the exporter's bytes, checked with ``tessera.serving``'s own readers."""
    out = _export(tmp_path, monkeypatch, _checkpoint(), {STACK: {"grid": "E4M3", "q256": 1024}})

    written = json.loads((out / "config.json").read_text())["quantization_config"]
    groups = [g for g in written["config_groups"].values() if g["targets"] == [STACK]]
    assert len(groups) == 1, sorted(written["config_groups"])
    scheme = groups[0]["scheme"]
    assert scheme["structure"] == "routed_moe", scheme
    assert STACK not in written["ignore"], (
        "a stack that is declared must not also be ignored; TesseraConfig refuses that overlap")
    assert not [i for i in written["ignore"] if i.startswith(f"{STACK}.")], written["ignore"]

    # The reader is the gate.
    declared = validate_tessera_moe_scheme(scheme, STACK)
    assert declared["experts"] == EXPERTS
    assert declared["hidden_size"] == HIDDEN and declared["intermediate_size"] == MOE_INTER
    assert declared["groups"]["w13"]["roles"] == [("gate_proj", MOE_INTER), ("up_proj", MOE_INTER)]
    assert declared["groups"]["w2"]["roles"] == [("down_proj", HIDDEN)]

    with safetensors_torch.safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        names = set(handle.keys())
        wires = {n: handle.get_tensor(n) for n in names if n.startswith(f"{STACK}.")}
    assert sorted(wires) == sorted(f"{STACK}.{e}.{p}.wire" for e in range(EXPERTS)
                                   for p in export.EXPERT_PROJECTIONS), sorted(wires)
    assert all(t.dtype == torch.uint8 for t in wires.values())

    # Every container is what its group's declaration promised, member by
    # member -- the same call ``moe_route`` makes at load.
    for group in MOE_GROUPS:
        declaration = declared["groups"][group]
        stride = declaration["wire_stride"]
        roles = expert_role_declarations(declaration)
        lengths = []
        for expert in range(EXPERTS):
            for role, projection in zip(roles, export.MOE_GROUP_PROJECTIONS[group]):
                blob = bytes(wires[f"{STACK}.{expert}.{projection}.wire"].tolist())
                lengths.append(len(blob))
                # ``parse_tessera_expert_blob`` returns ``[(role, unit)]`` --
                # the parsed unit artifact itself, whose geometry hangs off its
                # manifest.  (This read was ``parsed.unit.geometry`` and was
                # wrong; nothing caught it because the case is @cuda-gated and
                # the CPU suite skips it.  It is the check, not a formality:
                # the parse compares the wire's columns against the sidecar's.)
                (name, parsed), = parse_tessera_expert_blob(blob, role, STACK)
                assert name == projection
                assert parsed.manifest.geometry.columns == declaration["columns"]
        assert max(lengths) == stride, (
            f"group {group}: declared wire_stride {stride}, blobs max {max(lengths)}; the "
            "stride is the max over the group's blobs and unpack_moe_wires refuses anything else")

    # And the parameter layout the loader builds accepts them: pack the rows
    # the way ``create_weights`` + ``_load_wire`` do, then run the unpacker
    # whose refusals are the load-time integrity gate.
    w13_stride = declared["groups"]["w13"]["wire_stride"]
    w2_stride = declared["groups"]["w2"]["wire_stride"]
    w13 = torch.zeros(EXPERTS, 2, w13_stride, dtype=torch.uint8)
    w13_len = torch.zeros(EXPERTS, 2, dtype=torch.long)
    w2 = torch.zeros(EXPERTS, w2_stride, dtype=torch.uint8)
    w2_len = torch.zeros(EXPERTS, dtype=torch.long)
    for expert in range(EXPERTS):
        for index, projection in enumerate(export.MOE_GROUP_PROJECTIONS["w13"]):
            blob = wires[f"{STACK}.{expert}.{projection}.wire"]
            w13[expert, index, :blob.numel()] = blob
            w13_len[expert, index] = blob.numel()
        blob = wires[f"{STACK}.{expert}.down_proj.wire"]
        w2[expert, :blob.numel()] = blob
        w2_len[expert] = blob.numel()
    back13, back2 = unpack_moe_wires(MoePacked(w13_wire=w13, w13_wire_len=w13_len,
                                               w2_wire=w2, w2_wire_len=w2_len))
    assert len(back13) == EXPERTS and len(back2) == EXPERTS

    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert manifest["routed_moe"]["disposition"] == "quantized"
    assert manifest["routed_moe"]["quantized_stacks"] == [STACK]
    assert manifest["routed_moe"]["quantized_source_tensors"] == EXPERTS * 3
    assert STACK not in manifest["routed_moe"]["modules"]
    record = manifest["modules"][STACK]
    assert record["structure"] == "routed_moe"
    assert len(record["roles"]) == EXPERTS * 3
    assert record["wire_stride"]["w13"] == w13_stride


@cuda
def test_the_written_wires_decode_to_the_stock_expert_tile(tmp_path, monkeypatch):
    """The last hop: ``prepare_tessera_moe_experts`` on the exporter's own bytes.

    This is the function ``process_weights_after_loading`` calls, so what it
    returns here is what the runtime's fused-MoE kernel would be handed.
    """
    from tessera.serving.moe_route import prepare_tessera_moe_experts

    out = _export(tmp_path, monkeypatch, _checkpoint(), {STACK: {"grid": "E4M3", "q256": 1024}})
    written = json.loads((out / "config.json").read_text())["quantization_config"]
    scheme = next(g["scheme"] for g in written["config_groups"].values() if g["targets"] == [STACK])
    declared = validate_tessera_moe_scheme(scheme, STACK)

    with safetensors_torch.safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        def blob(expert, projection):
            return bytes(handle.get_tensor(f"{STACK}.{expert}.{projection}.wire").tolist())

        blobs = {"w13": [[blob(e, "gate_proj"), blob(e, "up_proj")] for e in range(EXPERTS)],
                 "w2": [[blob(e, "down_proj")] for e in range(EXPERTS)]}

    prepared = prepare_tessera_moe_experts(blobs, declared, STACK, device="cuda")

    assert tuple(prepared.w13_weight.shape) == (EXPERTS, 2 * MOE_INTER, HIDDEN)
    assert tuple(prepared.w2_weight.shape) == (EXPERTS, HIDDEN, MOE_INTER)
    assert prepared.w13_weight.dtype == torch.float8_e4m3fn
    assert tuple(prepared.w13_weight_scale.shape) == (EXPERTS, 2 * MOE_INTER, 1)
    assert tuple(prepared.w2_weight_scale.shape) == (EXPERTS, HIDDEN, 1)
    assert prepared.w13_weight_scale.dtype == torch.float32
    assert torch.isfinite(prepared.w13_weight_scale).all()
    assert (prepared.w13_weight_scale > 0).all()
