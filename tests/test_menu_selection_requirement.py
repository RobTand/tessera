"""The continuous Tessera menu must not ship surrogate-selected (issue #2).

Tessera#1 measured the failure: at matched bytes the surrogate-selected
allocation serves 2.00x worse KL than the uniform arm at 4.0 bpp, and 95% of
that gap sits on the seven units the surrogate itself priced
(``docs/measurements/tessera-allocated-served-2026-09-02.md`` §5, §7).  So a
plan whose Tessera units sit at more than one rung embodies a *selection*,
and the menu's recipe requires that selection to be validated-surrogate
(``docs/ARCHITECTURE.md`` §4.10) -- not a suggestion, a requirement.

What is pinned here is derived from the plan, never from a roster: the number
of distinct (grid, rung) pairs over the plan's own Tessera units decides
whether a selection happened, and ``tessera.control.REQUIRED_SELECTION_MODE``
-- the constant the owning module publishes -- decides what the requirement
is called.  A test that restated today's rung list would pass no matter what
the code did; these fail as soon as the rule stops being enforced.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tessera.control import (
    BF16,
    REQUIRED_SELECTION_MODE,
    SELECTION_SCHEMA,
    PlannedUnit,
    selection_requirement,
)

ROOT = Path(__file__).resolve().parents[1]


def units(rungs, grid="E4M3", rows=1024, columns=1024):
    return [
        PlannedUnit(f"m{i}.weight", grid, rung, rows, columns)
        for i, rung in enumerate(rungs)
    ]


def test_the_required_mode_is_validated_surrogate_not_a_suggestion():
    assert REQUIRED_SELECTION_MODE == "validated-surrogate"


def test_a_mixed_rung_plan_embodies_a_selection_and_is_unvalidated():
    block = selection_requirement(units([749, 934, 1083, 1083, 1107, 1107, 1006]))
    assert block["schema"] == SELECTION_SCHEMA
    assert block["mode_required"] == REQUIRED_SELECTION_MODE
    assert block["requires_validation"] is True
    assert block["validated"] is False
    assert "tessera#1" in block["reason"]


def test_a_uniform_plan_embodies_no_rung_selection():
    block = selection_requirement(units([1006] * 7))
    assert block["requires_validation"] is False
    assert block["validated"] is True


def test_a_plan_with_no_tessera_unit_has_no_rate_axis_to_select():
    block = selection_requirement(
        [PlannedUnit("m0.weight", BF16, None, 1024, 1024)]
    )
    assert block["requires_validation"] is False
    assert block["validated"] is True


def test_two_families_are_a_selection_even_at_one_rung_each():
    block = selection_requirement(
        units([1006, 1006], grid="E4M3")
        + units([896, 896], grid="E2M1x2")
    )
    assert block["requires_validation"] is True
    assert block["validated"] is False


def test_the_block_is_json_so_a_sidecar_can_carry_it():
    json.dumps(selection_requirement(units([749, 1006])))


# ------------------------------------------------- the converter stamps it


def _converter():
    spec = importlib.util.spec_from_file_location(
        "plan_from_layer_config", ROOT / "experiments" / "plan_from_layer_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mixed_config(layer=0):
    rungs = {"self_attn.q_proj": 1083, "self_attn.k_proj": 1083,
             "self_attn.v_proj": 1083, "self_attn.o_proj": 934,
             "mlp.gate_proj": 1107, "mlp.up_proj": 1107, "mlp.down_proj": 749}
    config = {}
    for role, rung in rungs.items():
        fmt = f"TESSERA_E4M3_K1_R{rung}"
        config[f"model.layers.{layer}.{role}"] = {
            "data_type": "tessera", "tessera_format": fmt,
            "tessera_family": "TESSERA_E4M3_K1",
            "tessera_body_rate_q256": rung}
    return config


def _shapes(layers=1):
    per_role = {"self_attn.q_proj": (2048, 1024), "self_attn.k_proj": (1024, 1024),
                "self_attn.v_proj": (1024, 1024), "self_attn.o_proj": (1024, 2048),
                "mlp.gate_proj": (3072, 1024), "mlp.up_proj": (3072, 1024),
                "mlp.down_proj": (1024, 3072)}
    return {f"model.layers.{n}.{role}.weight": shape
            for n in range(layers) for role, shape in per_role.items()}


def test_the_converter_sidecar_carries_the_selection_block():
    converter = _converter()
    _plan, provenance = converter.build(
        _mixed_config(), _shapes(), cover="as-allocated",
        allow_disagreement=False, prismaquant=None)
    block = provenance["selection"]
    assert block["mode_required"] == "validated-surrogate"
    assert block["requires_validation"] is True
    assert block["validated"] is False
    assert "tessera#1" in block["reason"]


def test_the_converter_sidecar_marks_a_uniform_plan_as_nothing_to_validate():
    converter = _converter()
    config = {qname: {"data_type": "tessera",
                      "tessera_format": "TESSERA_E4M3_K1_R1006",
                      "tessera_family": "TESSERA_E4M3_K1",
                      "tessera_body_rate_q256": 1006}
              for qname in _mixed_config()}
    _plan, provenance = converter.build(
        config, _shapes(), cover="as-allocated",
        allow_disagreement=False, prismaquant=None)
    assert provenance["selection"]["requires_validation"] is False


def test_the_converter_warns_that_a_mixed_plan_must_not_ship_unvalidated(
    tmp_path, capsys,
):
    import torch
    from safetensors.torch import save_file

    converter = _converter()
    model = tmp_path / "model"
    model.mkdir()
    tensors = {name: torch.zeros(4, 4) for name in _shapes()}
    save_file(tensors, model / "model.safetensors")
    layer_config = tmp_path / "lc.json"
    layer_config.write_text(json.dumps(_mixed_config()))
    out = tmp_path / "plan.json"
    assert converter.main([str(layer_config), str(model), str(out)]) == 0
    printed = capsys.readouterr().out
    assert "SELECTION" in printed and "validated-surrogate" in printed
    sidecar = json.loads(out.with_suffix(".json.provenance.json").read_text())
    assert sidecar["selection"]["requires_validation"] is True
