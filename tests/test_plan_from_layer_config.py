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
import box_artifacts

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plan_from_layer_config", ROOT / "experiments" / "plan_from_layer_config.py")
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)

PQ_TREE = box_artifacts.root("prismaquant_worktree")
ALLOC = box_artifacts.path("shared_runs", "pq-continuous", "qwen06b", "alloc")
MODEL = box_artifacts.path("models", "Qwen3-0.6B")


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

@box_artifacts.require("prismaquant_worktree", "prismaquant", "tessera_formats.py")
def test_the_sidecar_reproduces_prismaquants_own_charged_bits():
    # Not "a plausible size": the exact integer the allocator spent its budget
    # in.  Computed by importing PrismaQuant's own accountant, so a divergence
    # here is a divergence between two trees, not between two formulas.
    config = json.loads(box_artifacts.skip_now(
        "shared_runs", "pq-continuous", "qwen06b", "alloc", "lc_full_4.0.json").read_text())
    expected = config["__prismaquant__"]["body_assignment_payload_bits_total"]
    _plan, provenance = build(config, one_layer_shapes(), prismaquant=PQ_TREE)
    assert provenance["totals"]["prismaquant_charged_bits"] == expected
    assert provenance["totals"]["quantized_params"] == \
        config["__prismaquant__"]["body_assignment_quantizable_params"]
    assert provenance["totals"]["prismaquant_charged_bpp"] == pytest.approx(
        config["__prismaquant__"]["achieved_bits"], abs=1e-12)


@box_artifacts.require("prismaquant_worktree", "prismaquant", "tessera_formats.py")
@box_artifacts.require("shared_runs", "pq-continuous", "qwen06b", "alloc")
def test_broadcasting_keeps_the_allocations_bpp_because_every_layer_has_one_shape():
    config = json.loads((ALLOC / "lc_full_4.0.json").read_text())
    achieved = config["__prismaquant__"]["achieved_bits"]
    _plan, provenance = build(config, one_layer_shapes(layers=28),
                              cover="broadcast-by-role", prismaquant=PQ_TREE)
    assert provenance["totals"]["quantized_params"] == 28 * 15728640
    assert provenance["totals"]["prismaquant_charged_bpp"] == pytest.approx(achieved, abs=1e-12)


@box_artifacts.require("models", "Qwen3-0.6B")
def test_the_shapes_come_from_the_checkpoint_itself():
    shapes = PLAN.body_weights(MODEL)
    assert shapes == one_layer_shapes(layers=28)


# -- the fused roster is the exporter's, not a restatement (#211) --------------


def lfm_shapes():
    """An LFM dense MLP: ``feed_forward.w1/w3`` fuse into one ``w13`` Linear."""
    return {f"model.layers.0.feed_forward.{role}.weight": (128, 128)
            for role in ("w1", "w2", "w3")}


def test_the_fused_roster_is_derived_from_the_exporter():
    """Rule 3: the expected set comes from the code that owns it.

    ``export_tessera_serving.fused_module`` is the producer's one statement of
    which source leaves vLLM merges into one module.  The converter's
    ``fused_key`` restated two of its rows and missed the rest -- shared-expert
    gate/up and LFM ``w1/w3`` -> ``w13`` -- so plans over those groups skipped
    the fused invariant entirely (#211).  Derive the expectation from the
    exporter so a roster change there cannot silently strand this check again.
    """
    import export_tessera_serving as exporter

    for qname in ("model.layers.3.self_attn.q_proj",
                  "model.layers.3.self_attn.o_proj",
                  "model.layers.3.mlp.gate_proj",
                  "model.layers.3.mlp.down_proj",
                  "model.layers.3.mlp.shared_experts.up_proj",
                  "model.layers.3.feed_forward.w1",
                  "model.layers.3.feed_forward.w2",
                  "model.layers.3.feed_forward.w3",
                  "model.layers.3.mlp.experts.7.gate_proj"):
        expected = exporter.fused_module(qname + ".weight")
        got = PLAN.fused_key(qname)
        if expected is None:
            assert got is None, qname
        else:
            module, members = expected
            assert got == (module, tuple(m[: -len(".weight")] for m in members)), qname


def test_an_lfm_w1_tessera_w3_bf16_pair_is_refused_as_a_fused_disagreement():
    """The #211 converter repro: w1/w2 Tessera, w3 BF16, all 128x128.

    The serving exporter merges LFM's dense ``w1``/``w3`` into one ``w13``
    Linear and passes the whole group through when the members disagree, so a
    plan quantizing w1 and leaving w3 BF16 prices an encode that will not
    happen.  The converter's local roster did not know ``w13`` and built this
    plan with ``fused_disagreements == []``.
    """
    config = {"model.layers.0.feed_forward.w1": tessera("TESSERA_E4M3_K1_R1024"),
              "model.layers.0.feed_forward.w2": tessera("TESSERA_E4M3_K1_R1024"),
              "model.layers.0.feed_forward.w3": "BF16"}
    with pytest.raises(SystemExit, match="do not share one"):
        build(config, lfm_shapes(), with_control=False)


def test_the_lfm_disagreement_demotes_the_w13_group_under_the_override():
    config = {"model.layers.0.feed_forward.w1": tessera("TESSERA_E4M3_K1_R1024"),
              "model.layers.0.feed_forward.w2": tessera("TESSERA_E4M3_K1_R1024"),
              "model.layers.0.feed_forward.w3": "BF16"}
    plan, provenance = build(config, lfm_shapes(), allow_disagreement=True,
                             with_control=False)
    assert plan["model.layers.0.feed_forward.w1.weight"] == "BF16"
    assert plan["model.layers.0.feed_forward.w3.weight"] == "BF16"
    assert plan["model.layers.0.feed_forward.w2.weight"] == {"grid": "E4M3", "q256": 1024}
    assert [d["module"] for d in provenance["fused_disagreements"]] == \
           ["model.layers.0.feed_forward.w13"]
    entry = provenance["fused_disagreements"][0]
    assert entry["planned_as"] == "BF16"
    assert provenance["totals"]["demoted_to_bf16_params"] == \
        sum(entry["demoted_params"].values())


def test_demotion_accounting_counts_only_members_the_allocation_priced():
    """``totals.demoted_to_bf16_params`` is, by its own docstring, "params the
    allocation gave a Tessera rung and the plan gives BF16".  A member the
    allocation itself chose BF16 was never demoted -- the plan agrees with the
    allocation about it -- and counting it over-reports the one number whose
    job is to say how far the served allocation drifted from the chosen one.
    """
    config = {"model.layers.0.feed_forward.w1": tessera("TESSERA_E4M3_K1_R1024"),
              "model.layers.0.feed_forward.w2": tessera("TESSERA_E4M3_K1_R1024"),
              "model.layers.0.feed_forward.w3": "BF16"}
    _plan, provenance = build(config, lfm_shapes(), allow_disagreement=True,
                              with_control=False)
    entry = provenance["fused_disagreements"][0]
    assert entry["demoted_params"] == {"model.layers.0.feed_forward.w1": 128 * 128}
    assert provenance["totals"]["demoted_to_bf16_params"] == 128 * 128


def test_an_agreeing_lfm_pair_plans_member_by_member():
    """Differing rungs on one family are still not a disagreement (#37)."""
    config = {"model.layers.0.feed_forward.w1": tessera("TESSERA_E4M3_K1_R1024"),
              "model.layers.0.feed_forward.w2": tessera("TESSERA_E4M3_K1_R1024"),
              "model.layers.0.feed_forward.w3": tessera("TESSERA_E4M3_K1_R920")}
    plan, provenance = build(config, lfm_shapes(), with_control=False)
    assert provenance["fused_disagreements"] == []
    assert plan["model.layers.0.feed_forward.w1.weight"] == {"grid": "E4M3", "q256": 1024}
    assert plan["model.layers.0.feed_forward.w3.weight"] == {"grid": "E4M3", "q256": 920}


def test_shared_expert_gate_up_is_the_exporters_fused_group_too():
    """``mlp.shared_experts.gate_proj/up_proj`` merge exactly as the body MLP's do."""
    shapes = {f"model.layers.0.mlp.shared_experts.{role}.weight": (128, 128)
              for role in ("gate_proj", "up_proj", "down_proj")}
    config = {"model.layers.0.mlp.shared_experts.gate_proj": tessera("TESSERA_E4M3_K1_R1024"),
              "model.layers.0.mlp.shared_experts.up_proj": "BF16",
              "model.layers.0.mlp.shared_experts.down_proj": tessera("TESSERA_E4M3_K1_R1024")}
    with pytest.raises(SystemExit, match="do not share one"):
        build(config, shapes, with_control=False)


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


def test_shorthand_and_dictionary_preserve_mixed_dense_assignment():
    config = uniform_config()
    config["model.layers.0.mlp.down_proj"] = "BF16"
    expected = build(config, one_layer_shapes(), with_control=False)
    config = {name: entry["tessera_format"] if isinstance(entry, dict) else entry
              for name, entry in config.items()}
    assert build(config, one_layer_shapes(), with_control=False) == expected


@pytest.mark.parametrize("entry", ["TESSERA_E4M3_K1_R1024\n",
                                    {"tessera_format": "TESSERA_E4M3_K1_R1024\n"}])
def test_rung_grammar_refuses_trailing_newlines(entry):
    with pytest.raises(PLAN.PlanError, match="spelling"):
        PLAN.parse_entry("model.layers.0.mlp.down_proj", entry)


def _moe_plan_source(tmp_path, *, packed=False):
    import torch
    from safetensors.torch import save_file
    from tessera.serving_parts import source_identity
    import export_tessera_serving as export

    src = tmp_path / "source"
    src.mkdir()
    stack = "model.layers.0.feed_forward.experts"
    tensors = {f"model.layers.0.feed_forward.gate.weight": torch.zeros(2, 64),
               "model.layers.0.self_attn.o_proj.weight": torch.zeros(64, 64)}
    if packed:
        tensors.update({f"{stack}.gate_up_proj.weight": torch.zeros(2, 128, 64),
                        f"{stack}.down_proj.weight": torch.zeros(2, 64, 64)})
    else:
        tensors.update({f"{stack}.{expert}.{role}.weight": torch.zeros(64, 64)
                        for expert in range(2) for role in ("w1", "w2", "w3")})
    save_file(tensors, str(src / "model.safetensors"))
    config = {"architectures": ["Lfm2MoeForCausalLM"], "hidden_size": 64,
              "moe_intermediate_size": 64, "num_experts": 2}
    (src / "config.json").write_text(json.dumps(config))
    request = {stack: {"grid": "E4M3", "q256": 1024,
                      "source_layout": "out_first_chunked" if packed else "unpacked_per_expert"}}
    projection = export.project_expert_plan({n: tuple(t.shape) for n, t in tensors.items()}, config, request)
    projection["source"] = source_identity(src)
    keys = ("cols", "expert", "group", "projection", "rows", "source_layout",
            "source_slice", "source_tensor", "tensor")
    units = {u["tensor"][:-7]: {k: u[k] for k in keys}
             for u in projection["stacks"][stack]["units"]}
    carried = {"schema": "prismaquant.tessera_expert_projection.v1", "producer": projection,
               "stacks": {stack: units}, "request": request}
    return src, stack, units, carried


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("choice", ["TESSERA_E4M3_K1_R1024", "BF16"])
def test_actual_translator_hands_off_whole_expert_stacks(tmp_path, monkeypatch, packed, choice):
    import export_tessera_serving as export
    src, stack, units, carried = _moe_plan_source(tmp_path, packed=packed)
    assignment = {name: choice for name in units}
    router = "model.layers.0.feed_forward.gate"
    assignment[router] = "BF16"
    assignment["model.layers.0.self_attn.o_proj"] = "BF16"
    assignment["__prismaquant__"] = {"tessera_expert_projection": carried}
    path, out = tmp_path / "assignment.json", tmp_path / "plan.json"
    path.write_text(json.dumps(assignment))
    PLAN.main([str(path), str(src), str(out), "--no-uniform-control"])
    plan = json.loads(out.read_text())
    assert router + ".weight" not in plan
    assert not set(plan) & {n + ".weight" for n in units}
    assert set(plan) == {stack, "model.layers.0.self_attn.o_proj.weight"}
    provenance = json.loads(out.with_suffix(".json.provenance.json").read_text())
    assert provenance["coverage"]["unplanned_body_linears"] == 0
    dictionary = {name: tessera(entry) if isinstance(entry, str) and entry.startswith("TESSERA_") else entry
                  for name, entry in assignment.items()}
    path.write_text(json.dumps(dictionary))
    PLAN.main([str(path), str(src), str(out), "--no-uniform-control"])
    assert json.loads(out.read_text()) == plan
    assert json.loads(out.with_suffix(".json.provenance.json").read_text()) == provenance
    if not packed:
        dictionary.pop("__prismaquant__")
        path.write_text(json.dumps(dictionary))
        PLAN.main([str(path), str(src), str(out), "--no-uniform-control"])
        assert json.loads(out.read_text()) == plan
    if choice == "BF16":
        assert plan[stack] == "BF16"
        assert provenance["totals"]["quantized_params"] == 0
    else:
        assert plan[stack] == carried["request"][stack]
        _, dense, packed_shapes, routed = export.quantizable(src)
        actual = export.project_expert_plan({**dense, **packed_shapes, **routed},
                    json.loads((src / "config.json").read_text()), {stack: plan[stack]})
        assert actual["stacks"][stack]["units"] == carried["producer"]["stacks"][stack]["units"]
        assert provenance["totals"]["quantized_params"] == 6 * 64 * 64

    class PlanningCompleted(Exception):
        pass

    def after_planning(_config, targets):
        assert targets == ([stack] if choice != "BF16" else [])
        raise PlanningCompleted

    monkeypatch.setattr(export, "unrouted_modules", after_planning)
    argv = ["export", str(src), str(tmp_path / "export"), "--plan-json", str(out),
            "--grid", "E4M3", "--q256", "1024", "--device", "cpu"]
    if choice == "BF16":
        argv += ["--layers", "0"]  # explicitly request a pure passthrough copy
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(PlanningCompleted):
        export.main()


@pytest.mark.parametrize("bad_choice", ["BF16", "TESSERA_E4M3_K1_R1083"])
def test_translator_refuses_incompatible_projected_stack_before_export(tmp_path, bad_choice):
    src, stack, units, carried = _moe_plan_source(tmp_path)
    assignment = {name: "TESSERA_E4M3_K1_R1024" for name in units}
    assignment[next(iter(units))] = bad_choice
    assignment["__prismaquant__"] = {"tessera_expert_projection": carried}
    path, out = tmp_path / "assignment.json", tmp_path / "plan.json"
    path.write_text(json.dumps(assignment))
    with pytest.raises(PLAN.PlanError, match="whole stack"):
        PLAN.main([str(path), str(src), str(out), "--no-uniform-control"])
    assert not out.exists()
