"""The PrismaQuant ``layer_config.json`` -> Tessera ``--plan-json`` converter.

The converter is the only translation between an allocator's currency (qnames
and ``TESSERA_<BASE>_K<arity>_R<rung>``) and the exporter's (tensor names and
``{"grid", "q256"}``), so what it must not do is quietly change the assignment:
drop a unit, invent a rate, or write a plan whose fused groups the serving path
cannot express.  Each test below pins one of those.

Differing RUNGS inside one fused group are NOT such a case (#37): the serving
path decodes each role from its own manifest and the exporter writes a per-role
``q256`` list, so a mink allocation is planned member by member.  What a fused
group must still share is its ROUTE -- family, grid, body, scale plane -- and
that is one imported key (``export_tessera_serving.module_scheme_key``) rather
than a rule this file states a second time.
"""
from __future__ import annotations

import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plan_from_layer_config", ROOT / "experiments" / "plan_from_layer_config.py")
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)

PQ_TREE = Path("/home/rob/pq-wt/tessera-continuous")
ALLOC = Path("/mnt/shared/tessera-runs/pq-continuous/qwen06b/alloc")
MODEL = Path("/home/rob/models/Qwen3-0.6B")


def tessera(fmt: str) -> dict:
    family, rung = fmt.rsplit("_R", 1)
    return {"data_type": "tessera", "tessera_format": fmt, "tessera_family": family,
            "tessera_body_rate_q256": int(rung), "tessera_body": "WINDOW",
            "tessera_scale_plane": "channel"}


def one_layer_shapes(layers=1):
    """Qwen3-0.6B's seven body Linears per decoder layer."""
    per_role = {"self_attn.q_proj": (2048, 1024), "self_attn.k_proj": (1024, 1024),
                "self_attn.v_proj": (1024, 1024), "self_attn.o_proj": (1024, 2048),
                "mlp.gate_proj": (3072, 1024), "mlp.up_proj": (3072, 1024),
                "mlp.down_proj": (1024, 3072)}
    return {f"model.layers.{n}.{role}.weight": shape
            for n in range(layers) for role, shape in per_role.items()}


def uniform_config(fmt="TESSERA_E4M3_K1_R1083", layer=0):
    return {f"model.layers.{layer}.{role}": tessera(fmt)
            for role in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}


def build(config, shapes, **kw):
    kw.setdefault("cover", "as-allocated")
    kw.setdefault("allow_disagreement", False)
    kw.setdefault("prismaquant", None)
    return PLAN.build(config, shapes, **kw)


# -- the name translation ------------------------------------------------------

def test_the_family_spelling_maps_to_the_exporters_grid_vocabulary():
    # Arity is an ``xN`` suffix on one side and a ``_K<n>`` infix on the other;
    # a converter that got this wrong would export the wrong FAMILY, which is a
    # different route and a different activation contract, not a different rate.
    assert PLAN.grid_of("TESSERA_E4M3_K1") == "E4M3"
    assert PLAN.grid_of("TESSERA_E2M1_K1") == "E2M1"
    assert PLAN.grid_of("TESSERA_E2M1_K2") == "E2M1x2"
    with pytest.raises(SystemExit):
        PLAN.grid_of("NVFP4")


def test_the_rung_comes_from_the_format_name_and_the_sidecar_fields_must_agree():
    entry = tessera("TESSERA_E4M3_K1_R1083")
    assert PLAN.parse_entry("m", entry) == ("tessera", ("E4M3", 1083, "TESSERA_E4M3_K1"))
    entry["tessera_body_rate_q256"] = 1084
    with pytest.raises(SystemExit, match="disagrees"):
        PLAN.parse_entry("m", entry)
    entry = tessera("TESSERA_E4M3_K1_R1083")
    entry["tessera_family"] = "TESSERA_E2M1_K2"
    with pytest.raises(SystemExit, match="disagrees"):
        PLAN.parse_entry("m", entry)


def test_a_plan_names_tensors_not_qnames():
    plan, _ = build(uniform_config(), one_layer_shapes())
    assert set(plan) == set(one_layer_shapes())
    assert plan["model.layers.0.mlp.down_proj.weight"] == {"grid": "E4M3", "q256": 1083}


# -- what it refuses -----------------------------------------------------------

def test_a_non_tessera_quantised_choice_is_refused_by_name_and_count():
    # One checkpoint, one plugin: an NVFP4 unit has no route in tessera.serving,
    # and the exporter's own ``unknown`` check would not catch it (the tensor IS
    # a body Linear), so it would silently be exported at the default rung.
    config = uniform_config()
    config["model.layers.0.mlp.down_proj"] = {"data_type": "nvfp4", "bits": 4}
    config["model.layers.0.self_attn.o_proj"] = {"data_type": "fp8_dynamic", "bits": 8}
    with pytest.raises(SystemExit) as excinfo:
        build(config, one_layer_shapes())
    message = str(excinfo.value)
    assert "2 unit(s)" in message and "nvfp4" in message and "fp8_dynamic" in message


def test_a_bf16_choice_is_a_bf16_module_and_not_a_refusal():
    config = uniform_config()
    config["model.layers.0.mlp.down_proj"] = "BF16"
    plan, provenance = build(config, one_layer_shapes())
    assert plan["model.layers.0.mlp.down_proj.weight"] == "BF16"
    assert provenance["coverage"]["planned_bf16_units"] == 1
    assert provenance["totals"]["tessera_units"] == 6


def test_a_fused_group_whose_members_took_different_rungs_is_planned_as_allocated():
    """Differing RUNGS are not a disagreement (#37).

    vLLM builds ONE method per fused module, but that method decodes each role
    from that role's OWN manifest -- proved element-for-element by
    ``experiments/fused_member_rung_identity.py`` and published as a value by
    ``runtime_contract.json``'s ``fused_module.fields`` (``q256: per_member``).
    So the converter carries each member's rung through, and the exporter writes
    them as a per-role ``q256`` list in the module's scheme.  It used to refuse
    this, and the refusal cost the allocator's chosen point its export.
    """
    config = uniform_config()
    config["model.layers.0.self_attn.k_proj"] = tessera("TESSERA_E4M3_K1_R920")
    plan, provenance = build(config, one_layer_shapes())
    assert plan["model.layers.0.self_attn.q_proj.weight"] == {"grid": "E4M3", "q256": 1083}
    assert plan["model.layers.0.self_attn.k_proj.weight"] == {"grid": "E4M3", "q256": 920}
    assert plan["model.layers.0.self_attn.v_proj.weight"] == {"grid": "E4M3", "q256": 1083}
    assert provenance["fused_disagreements"] == []
    assert provenance["fused_disagreement_policy"] == "none"
    totals = provenance["totals"]
    assert totals["tessera_units"] == 7
    assert totals["demoted_to_bf16_params"] == 0
    assert totals["quantized_params"] == sum(r * c for r, c in one_layer_shapes().values())


def test_a_mink_allocation_plans_every_member_at_the_rung_the_dp_chose():
    """A per-member (mink) allocation, end to end: Tessera issues #15 and #37.

    PrismaQuant's group knapsack gives one family per fused group and a RATE
    PER MEMBER, which is the whole reason the option exists -- the members'
    sensitivities differ.  No single rate for the group is derivable from the
    objective (min / bytes-weighted / max are taste, not arithmetic), and the
    converter never needs one: it plans each member at its own rung, and #15's
    rule still holds -- every unit the sidecar prices is a unit that will be
    encoded, and nothing is charged for that will not serve.
    """
    config = {
        "model.layers.0.self_attn.q_proj": tessera("TESSERA_E4M3_K1_R1083"),
        "model.layers.0.self_attn.k_proj": tessera("TESSERA_E4M3_K1_R920"),
        "model.layers.0.self_attn.v_proj": tessera("TESSERA_E4M3_K1_R1200"),
        "model.layers.0.self_attn.o_proj": tessera("TESSERA_E4M3_K1_R934"),
        "model.layers.0.mlp.gate_proj": tessera("TESSERA_E4M3_K1_R1107"),
        "model.layers.0.mlp.up_proj": tessera("TESSERA_E4M3_K1_R1000"),
        "model.layers.0.mlp.down_proj": tessera("TESSERA_E4M3_K1_R749"),
    }
    plan, provenance = build(config, one_layer_shapes())
    assert {name: entry["q256"] for name, entry in plan.items()} == {
        "model.layers.0.self_attn.q_proj.weight": 1083,
        "model.layers.0.self_attn.k_proj.weight": 920,
        "model.layers.0.self_attn.v_proj.weight": 1200,
        "model.layers.0.self_attn.o_proj.weight": 934,
        "model.layers.0.mlp.gate_proj.weight": 1107,
        "model.layers.0.mlp.up_proj.weight": 1000,
        "model.layers.0.mlp.down_proj.weight": 749,
    }
    totals = provenance["totals"]
    assert provenance["fused_disagreements"] == []
    assert totals["tessera_units"] == 7
    assert totals["demoted_to_bf16_params"] == 0
    assert totals["quantized_params"] == sum(r * c for r, c in one_layer_shapes().values())
    # every priced unit is a unit that will be encoded (#15)
    assert {u["qname"] for u in provenance["units"]} == set(config)


def test_a_fused_group_split_across_two_families_is_still_refused_and_still_demoted():
    """What the relaxation did NOT relax.

    A rate is per role; a ROUTE is not.  Two members on two families would need
    two decoders to produce one tile, so the refusal stands, and
    ``--allow-fused-disagreement`` still writes the plan that will SERVE --
    every member BF16, dropped from the unit table, and the demotion recorded --
    rather than naming rungs the exporter is about to discard.
    """
    config = uniform_config()
    config["model.layers.0.self_attn.k_proj"] = tessera("TESSERA_BF16_K1_R1792")
    with pytest.raises(SystemExit, match="do not share one"):
        build(config, one_layer_shapes())
    plan, provenance = build(config, one_layer_shapes(), allow_disagreement=True)
    assert plan["model.layers.0.self_attn.k_proj.weight"] == "BF16"
    assert plan["model.layers.0.self_attn.q_proj.weight"] == "BF16"
    assert plan["model.layers.0.self_attn.v_proj.weight"] == "BF16"
    assert plan["model.layers.0.mlp.gate_proj.weight"] == {"grid": "E4M3", "q256": 1083}
    assert [d["module"] for d in provenance["fused_disagreements"]] == \
           ["model.layers.0.self_attn.qkv_proj"]
    entry = provenance["fused_disagreements"][0]
    assert entry["planned_as"] == "BF16"
    assert entry["demoted_params"] == {
        "model.layers.0.self_attn.q_proj": 2048 * 1024,
        "model.layers.0.self_attn.k_proj": 1024 * 1024,
        "model.layers.0.self_attn.v_proj": 1024 * 1024,
    }
    totals = provenance["totals"]
    assert totals["tessera_units"] == 4
    assert totals["demoted_to_bf16_params"] == 4 * 1024 * 1024
    assert totals["quantized_params"] == 3 * 3072 * 1024 + 1024 * 2048
    assert provenance["fused_disagreement_policy"] == \
        "demoted_to_bf16_by_--allow-fused-disagreement"


def test_a_whole_group_option_name_is_refused_as_the_thing_it_is():
    """``TESSERA_E4M3_K1_G3`` is a group option, not a rung, and no rate stands for it.

    PrismaQuant's ``expand_fused_sibling_assignment`` replaces it with the
    members' own rungs before an assignment is written; one that reaches a plan
    means that expansion did not run, and the useful error says so rather than
    complaining about a spelling.
    """
    config = uniform_config()
    config["model.layers.0.self_attn.k_proj"] = {
        "data_type": "tessera", "tessera_format": "TESSERA_E4M3_K1_G3",
        "tessera_family": "TESSERA_E4M3_K1",
    }
    with pytest.raises(SystemExit, match="whole-GROUP option"):
        build(config, one_layer_shapes())


def test_a_fused_group_whose_members_take_two_families_is_the_same_refusal():
    config = uniform_config()
    config["model.layers.0.mlp.up_proj"] = tessera("TESSERA_E2M1_K2_R896")
    with pytest.raises(SystemExit, match="do not share one"):
        build(config, one_layer_shapes())


def test_a_unit_the_model_does_not_carry_is_refused():
    config = uniform_config()
    config["model.layers.9.mlp.down_proj"] = tessera("TESSERA_E4M3_K1_R749")
    with pytest.raises(SystemExit, match="not a 2-D body Linear"):
        build(config, one_layer_shapes())


# -- coverage ------------------------------------------------------------------

def test_as_allocated_names_every_unpriced_linear_bf16_rather_than_leaving_it_out():
    # A tensor the plan does not name does NOT come out BF16: the exporter
    # falls back to its own --grid/--q256 default, which is a 4-bit NVFP4 rung.
    # A seven-unit plan that stayed silent about the other 189 therefore built
    # a 4-bit checkpoint nobody priced -- and, weights-only, one the exporter
    # refuses outright for want of --input-scales.  Silence is the bug; the
    # plan says BF16 out loud.
    plan, provenance = build(uniform_config(), one_layer_shapes(layers=28))
    assert len(plan) == 196
    assert sum(1 for v in plan.values() if v == "BF16") == 196 - 7
    assert plan["model.layers.0.self_attn.q_proj.weight"] == {"grid": "E4M3", "q256": 1083}
    assert plan["model.layers.27.self_attn.q_proj.weight"] == "BF16"
    coverage = provenance["coverage"]
    assert coverage["extrapolated"] is False
    assert coverage["unplanned_body_linears"] == 0


def test_broadcast_names_a_role_the_allocation_never_priced_bf16():
    # Same trap one level down: a role missing from the allocation entirely
    # (not chosen BF16, just absent) must still be named, or it silently takes
    # the exporter's 4-bit default at all 28 depths.
    config = uniform_config()
    del config["model.layers.0.mlp.down_proj"]
    plan, provenance = build(config, one_layer_shapes(layers=28), cover="broadcast-by-role")
    assert len(plan) == 196
    assert plan["model.layers.5.mlp.down_proj.weight"] == "BF16"
    assert provenance["coverage"]["unplanned_body_linears"] == 0


def test_broadcast_applies_the_per_role_assignment_at_every_depth_and_says_it_is_one():
    config = {"model.layers.0.self_attn.q_proj": tessera("TESSERA_E4M3_K1_R1083"),
              "model.layers.0.self_attn.k_proj": tessera("TESSERA_E4M3_K1_R1083"),
              "model.layers.0.self_attn.v_proj": tessera("TESSERA_E4M3_K1_R1083"),
              "model.layers.0.self_attn.o_proj": tessera("TESSERA_E4M3_K1_R934"),
              "model.layers.0.mlp.gate_proj": tessera("TESSERA_E4M3_K1_R1107"),
              "model.layers.0.mlp.up_proj": tessera("TESSERA_E4M3_K1_R1107"),
              "model.layers.0.mlp.down_proj": tessera("TESSERA_E4M3_K1_R749")}
    plan, provenance = build(config, one_layer_shapes(layers=28), cover="broadcast-by-role")
    assert len(plan) == 196
    assert plan["model.layers.27.mlp.down_proj.weight"] == {"grid": "E4M3", "q256": 749}
    assert plan["model.layers.13.self_attn.o_proj.weight"] == {"grid": "E4M3", "q256": 934}
    coverage = provenance["coverage"]
    assert coverage["extrapolated"] is True and coverage["broadcast_from_layer"] == 0
    assert coverage["unplanned_body_linears"] == 0


def test_broadcast_refuses_a_multi_layer_allocation():
    config = dict(uniform_config(layer=0), **uniform_config(layer=1))
    with pytest.raises(SystemExit, match="single-layer allocation"):
        build(config, one_layer_shapes(layers=2), cover="broadcast-by-role")


def test_broadcast_refuses_a_role_whose_shape_differs_at_another_depth():
    # A rung is a rate on a SHAPE: the CHANNEL plane amortises over rows, so the
    # same q256 on a different row count is a different wire bpp.  Broadcasting
    # across it would silently change the byte budget.
    shapes = one_layer_shapes(layers=2)
    shapes["model.layers.1.mlp.down_proj.weight"] = (2048, 3072)
    with pytest.raises(SystemExit, match="different rate"):
        build(uniform_config(), shapes, cover="broadcast-by-role")


# -- the accounting the export is checked against ------------------------------

@pytest.mark.skipif(not (PQ_TREE / "prismaquant" / "tessera_formats.py").exists(),
                    reason="the PrismaQuant tree with tessera_formats is not on this box")
def test_the_sidecar_reproduces_prismaquants_own_charged_bits():
    # Not "a plausible size": the exact integer the allocator spent its budget
    # in.  Computed by importing PrismaQuant's own accountant, so a divergence
    # here is a divergence between two trees, not between two formulas.
    config = json.loads((ALLOC / "lc_full_4.0.json").read_text()) \
        if ALLOC.exists() else None
    if config is None:
        pytest.skip("the allocation outputs are not on this box")
    expected = config["__prismaquant__"]["body_assignment_payload_bits_total"]
    _plan, provenance = build(config, one_layer_shapes(), prismaquant=PQ_TREE)
    assert provenance["totals"]["prismaquant_charged_bits"] == expected
    assert provenance["totals"]["quantized_params"] == \
        config["__prismaquant__"]["body_assignment_quantizable_params"]
    assert provenance["totals"]["prismaquant_charged_bpp"] == pytest.approx(
        config["__prismaquant__"]["achieved_bits"], abs=1e-12)


@pytest.mark.skipif(not (PQ_TREE / "prismaquant" / "tessera_formats.py").exists()
                    or not ALLOC.exists(),
                    reason="the PrismaQuant tree or the allocation outputs are not on this box")
def test_broadcasting_keeps_the_allocations_bpp_because_every_layer_has_one_shape():
    config = json.loads((ALLOC / "lc_full_4.0.json").read_text())
    achieved = config["__prismaquant__"]["achieved_bits"]
    _plan, provenance = build(config, one_layer_shapes(layers=28),
                              cover="broadcast-by-role", prismaquant=PQ_TREE)
    assert provenance["totals"]["quantized_params"] == 28 * 15728640
    assert provenance["totals"]["prismaquant_charged_bpp"] == pytest.approx(achieved, abs=1e-12)


@pytest.mark.skipif(not MODEL.exists(), reason="Qwen3-0.6B is not on this box")
def test_the_shapes_come_from_the_checkpoint_itself():
    shapes = PLAN.body_weights(MODEL)
    assert shapes == one_layer_shapes(layers=28)


def test_the_unit_table_carries_the_shape_the_rate_was_charged_on():
    _plan, provenance = build(uniform_config(), one_layer_shapes(), prismaquant=PQ_TREE)
    rows = {u["qname"]: u for u in provenance["units"]}
    assert rows["model.layers.0.self_attn.q_proj"]["rows"] == 2048
    assert rows["model.layers.0.self_attn.k_proj"]["rows"] == 1024
    if rows["model.layers.0.self_attn.q_proj"]["prismaquant_charged_bpp"] is not None:
        # §8.2 of the PrismaQuant receipt: the same rung is a different wire bpp
        # on a different row count, because the CHANNEL plane amortises over rows.
        assert (rows["model.layers.0.self_attn.q_proj"]["prismaquant_charged_bpp"]
                < rows["model.layers.0.self_attn.k_proj"]["prismaquant_charged_bpp"])
