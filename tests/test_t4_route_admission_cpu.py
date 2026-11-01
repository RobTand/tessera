"""CPU proof for T-4 route admission.

Every check runs on the CPU. No test allocates CUDA, imports vLLM,
or loads a native extension. Each pass case proves admission. Each
refusal case proves the fail-closed name.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("torch")

from tessera.t4_route_admission import (
    T4_PURE_Q256,
    build_attestation_stub,
    build_preflight,
    expected_pairs,
    require_census_pair,
    require_dense_geometry,
    require_execution_mode,
    require_pure_q256,
    require_routed_geometry,
    require_routed_rates,
    require_routed_shape_agreement,
    require_served_recipe,
    require_structure,
    t4_cell_agreement,
    t4_census_expected,
)


def _recipe(body="WINDOW", span=1, plane="LUT", window_bits=14):
    return types.SimpleNamespace(
        body=body, span=span, scale_plane=plane, window_bits=window_bits
    )


@pytest.mark.parametrize("q256", list(T4_PURE_Q256))
def test_pure_classes_pass(q256):
    assert require_pure_q256(q256) == q256


@pytest.mark.parametrize("q256", [0, 127, 129, 1152, "896", None, 896.0, -128])
def test_impure_rungs_refuse(q256):
    with pytest.raises(ValueError, match="t4_admission: impure rung"):
        require_pure_q256(q256)


@pytest.mark.parametrize("structure", ["dense", "routed_moe"])
def test_structures_pass(structure):
    assert require_structure(structure) == structure


@pytest.mark.parametrize("structure", ["moe", "", None, "DENSE", 0])
def test_structures_refuse(structure):
    with pytest.raises(ValueError, match="t4_admission: unknown structure"):
        require_structure(structure)


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_modes_pass(mode):
    assert require_execution_mode(mode) == mode


@pytest.mark.parametrize("mode", ["compiled", "", None, "EAGER", 0])
def test_modes_refuse(mode):
    with pytest.raises(ValueError, match="t4_admission: unknown execution mode"):
        require_execution_mode(mode)


def test_served_recipe_passes():
    assert require_served_recipe(_recipe()) is not None


def test_served_recipe_dict_passes():
    recipe = {"body": "WINDOW", "span": 1, "scale_plane": "LUT", "window_bits": 14}
    assert require_served_recipe(recipe) is not None


def test_real_served_recipe_passes():
    from tessera.export import E2M1X2_SERVED_RECIPE

    assert require_served_recipe(E2M1X2_SERVED_RECIPE) is not None


@pytest.mark.parametrize(
    "recipe,match",
    [
        (_recipe(body="TCQ"), "t4_admission: served recipe body"),
        (_recipe(span=2), "t4_admission: served recipe span"),
        (_recipe(plane="CHANNEL"), "t4_admission: served recipe plane"),
        (_recipe(window_bits=12), "t4_admission: served recipe window_bits"),
        ({}, "t4_admission: served recipe lacks"),
    ],
)
def test_served_recipe_refuses(recipe, match):
    with pytest.raises(ValueError, match=match):
        require_served_recipe(recipe)


@pytest.mark.parametrize(
    "rows,cols", [(32, 256), (64, 512), (128, 256), (256, 4096)]
)
def test_dense_geometries_pass(rows, cols):
    require_dense_geometry(rows, cols)


@pytest.mark.parametrize(
    "rows,cols", [(31, 256), (32, 128), (32, 288), (0, 256), (-32, 256)]
)
def test_dense_geometries_refuse(rows, cols):
    with pytest.raises(ValueError, match="t4_admission: dense geometry refused"):
        require_dense_geometry(rows, cols)


@pytest.mark.parametrize(
    "rows,cols,projection",
    [(128, 256, "gate_proj"), (128, 512, "up_proj"), (256, 256, "down_proj")],
)
def test_routed_geometries_pass(rows, cols, projection):
    require_routed_geometry(rows, cols, projection)


@pytest.mark.parametrize(
    "rows,cols,projection",
    [
        (127, 256, "gate_proj"),
        (128, 128, "up_proj"),
        (128, 256, "down"),
        (128, 256, None),
    ],
)
def test_routed_geometries_refuse(rows, cols, projection):
    with pytest.raises(ValueError, match="t4_admission:"):
        require_routed_geometry(rows, cols, projection)


@pytest.mark.parametrize("q256", list(T4_PURE_Q256))
def test_routed_rates_pass(q256):
    assert require_routed_rates([[q256, q256, q256]]) == [[q256, q256, q256]]


def test_routed_rates_refuse_impure():
    with pytest.raises(ValueError, match="t4_admission: impure routed rung"):
        require_routed_rates([[896, 897, 896]])


def test_routed_rates_refuse_disagreeing_experts():
    with pytest.raises(ValueError, match="t4_admission: routed rates refused"):
        require_routed_rates([[896, 896, 896], [768, 768, 768]])


def test_routed_rates_refuse_gate_up_split():
    with pytest.raises(ValueError, match="t4_admission: routed rates refused"):
        require_routed_rates([[896, 768, 896]])


@pytest.mark.parametrize("rungs", [[], [[]], [[896], [768, 768]], "896", None])
def test_routed_rates_refuse_shape(rungs):
    with pytest.raises(ValueError, match="t4_admission:"):
        require_routed_rates(rungs)


def test_census_expected_has_t4_pairs():
    from tessera.serving.scheme import (
        FUSED_WINDOW_DENSE_E2M1_SYMBOL,
        ROUTED_FUSED_WINDOW_E2M1_SYMBOL,
    )
    from tessera.serving.telemetry import (
        DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1,
        DECODER_NATIVE_ROUTED_FUSED_WINDOW_E2M1,
    )

    dense = t4_census_expected("dense")
    routed = t4_census_expected("routed_moe")
    assert (FUSED_WINDOW_DENSE_E2M1_SYMBOL, DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1) in dense
    assert (
        ROUTED_FUSED_WINDOW_E2M1_SYMBOL,
        DECODER_NATIVE_ROUTED_FUSED_WINDOW_E2M1,
    ) in routed
    assert dense == expected_pairs("dense")
    assert t4_census_expected("dense", "decode")
    assert t4_census_expected("dense", "batch")


def test_census_expected_refuses():
    with pytest.raises(ValueError, match="t4_admission: unknown structure"):
        t4_census_expected("moe")
    with pytest.raises(ValueError, match="t4_admission: unknown regime"):
        t4_census_expected("dense", "prefill")


def test_require_census_pair_passes():
    from tessera.serving.scheme import FUSED_WINDOW_DENSE_E2M1_SYMBOL
    from tessera.serving.telemetry import DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1

    assert require_census_pair(
        "dense", FUSED_WINDOW_DENSE_E2M1_SYMBOL, DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1
    )


def test_require_census_pair_refuses():
    with pytest.raises(ValueError, match="t4_admission: unexpected census pair"):
        require_census_pair("dense", "no.symbol", "no_decoder")


def test_cell_agreement():
    pairs = {("a", "b")}
    assert t4_cell_agreement({"symbol": "a", "decoder": "b"}, pairs) == (True, None)
    agrees, problem = t4_cell_agreement({"symbol": "a", "decoder": "c"}, pairs)
    assert agrees is False and "t4_admission:" in problem
    agrees, problem = t4_cell_agreement({"symbol": "a"}, pairs)
    assert agrees is False and "t4_admission:" in problem
    with pytest.raises(ValueError, match="t4_admission: census record must be"):
        t4_cell_agreement("a", pairs)


@pytest.mark.parametrize("structure", ["dense", "routed_moe"])
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_attestation_stays_unmeasured(structure, mode):
    stub = build_attestation_stub({"structure": structure, "q256": 896}, mode)
    assert stub["status"] == "not_measured"
    assert stub["gpu_executed"] is False
    assert stub["qualification"] == "not_measured"
    assert stub["serving"] == "not_attested"
    assert stub["execution_mode"] == mode


def _dense_scope(q256, mode):
    from tessera.serving.scheme import FUSED_WINDOW_DENSE_E2M1_SYMBOL
    from tessera.serving.telemetry import DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1

    return {
        "structure": "dense",
        "execution_mode": mode,
        "q256": q256,
        "rows": 32,
        "columns": 256,
        "recipe": _recipe(),
        "symbol": FUSED_WINDOW_DENSE_E2M1_SYMBOL,
        "decoder": DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1,
    }


def _routed_scope(q256, mode):
    from tessera.serving.scheme import ROUTED_FUSED_WINDOW_E2M1_SYMBOL
    from tessera.serving.telemetry import DECODER_NATIVE_ROUTED_FUSED_WINDOW_E2M1

    return {
        "structure": "routed_moe",
        "execution_mode": mode,
        "rungs": [[q256, q256, q256]],
        "shapes": {
            "gate_proj": (2048, 4096),
            "up_proj": (2048, 4096),
            "down_proj": (4096, 2048),
        },
        "recipe": _recipe(),
        "symbol": ROUTED_FUSED_WINDOW_E2M1_SYMBOL,
        "decoder": DECODER_NATIVE_ROUTED_FUSED_WINDOW_E2M1,
    }


@pytest.mark.parametrize("q256", list(T4_PURE_Q256))
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_dense_preflight_passes(q256, mode):
    receipt = build_preflight(_dense_scope(q256, mode))
    assert receipt["status"] == "admitted_cpu"
    assert receipt["gpu_executed"] is False
    assert receipt["qualification"] == "not_measured"
    assert receipt["serving"] == "not_attested"


@pytest.mark.parametrize("q256", list(T4_PURE_Q256))
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_routed_preflight_passes(q256, mode):
    receipt = build_preflight(_routed_scope(q256, mode))
    assert receipt["status"] == "admitted_cpu"
    assert receipt["gpu_executed"] is False
    assert receipt["qualification"] == "not_measured"
    assert receipt["serving"] == "not_attested"


def test_preflight_refuses_impure_dense():
    scope = _dense_scope(897, "eager")
    with pytest.raises(ValueError, match="t4_admission: impure rung"):
        build_preflight(scope)


def test_preflight_refuses_bad_dense_geometry():
    scope = _dense_scope(896, "eager")
    scope["rows"] = 31
    with pytest.raises(ValueError, match="t4_admission: dense geometry refused"):
        build_preflight(scope)


def test_preflight_refuses_bad_pair():
    scope = _dense_scope(896, "eager")
    scope["decoder"] = "no_decoder"
    with pytest.raises(ValueError, match="t4_admission: unexpected census pair"):
        build_preflight(scope)


def test_preflight_refuses_routed_without_shapes():
    scope = _routed_scope(896, "graph")
    del scope["shapes"]
    with pytest.raises(ValueError, match="t4_admission: routed scope needs"):
        build_preflight(scope)


def test_preflight_refuses_unknown_mode():
    scope = _dense_scope(896, "compiled")
    with pytest.raises(ValueError, match="t4_admission: unknown execution mode"):
        build_preflight(scope)


@pytest.mark.parametrize("rows,cols", [(32.5, 256), (32.0, 256), ("32", 256), (True, 256)])
def test_dense_geometries_refuse_fractional(rows, cols):
    with pytest.raises(ValueError, match="t4_admission: dense shape must be integers"):
        require_dense_geometry(rows, cols)


@pytest.mark.parametrize(
    "rows,cols,projection",
    [(128.5, 256, "gate_proj"), (128, 256.0, "up_proj"), ("256", 256, "down_proj")],
)
def test_routed_geometries_refuse_fractional(rows, cols, projection):
    with pytest.raises(ValueError, match="t4_admission: routed shape must be integers"):
        require_routed_geometry(rows, cols, projection)


@pytest.mark.parametrize("span", [1.0, True, "1"])
def test_served_recipe_refuses_fractional_span(span):
    with pytest.raises(ValueError, match="t4_admission: served recipe span"):
        require_served_recipe(_recipe(span=span))


@pytest.mark.parametrize("window_bits", [14.0, "14"])
def test_served_recipe_refuses_fractional_width(window_bits):
    with pytest.raises(ValueError, match="t4_admission: served recipe window_bits"):
        require_served_recipe(_recipe(window_bits=window_bits))


@pytest.mark.parametrize("rungs", [[[896]], [[896, 896]], [[896] * 4]])
def test_routed_rates_refuse_non_triple(rungs):
    with pytest.raises(ValueError, match="t4_admission: routed rates need exactly three"):
        require_routed_rates(rungs)


def test_preflight_refuses_routed_row_disagreement():
    scope = _routed_scope(896, "eager")
    scope["shapes"]["up_proj"] = (1024, 4096)
    with pytest.raises(ValueError, match="t4_admission: routed rows disagree"):
        build_preflight(scope)


def test_preflight_refuses_routed_column_disagreement():
    scope = _routed_scope(896, "eager")
    scope["shapes"]["down_proj"] = (2048, 2048)
    with pytest.raises(ValueError, match="t4_admission: routed columns disagree"):
        build_preflight(scope)


def test_preflight_refuses_routed_fractional_shape():
    scope = _routed_scope(896, "graph")
    scope["shapes"]["gate_proj"] = (2048.0, 4096)
    with pytest.raises(ValueError, match="t4_admission: gate_proj shape must be two integers"):
        build_preflight(scope)


def _agree_shapes():
    return {
        "gate_proj": (2048, 4096),
        "up_proj": (2048, 4096),
        "down_proj": (4096, 2048),
    }


def test_routed_agreement_passes():
    assert require_routed_shape_agreement(_agree_shapes()) == _agree_shapes()


@pytest.mark.parametrize(
    "shapes,match",
    [
        (
            {"gate_proj": (2048, 4096), "up_proj": (1024, 4096), "down_proj": (4096, 2048)},
            "t4_admission: routed rows disagree",
        ),
        (
            {"gate_proj": (2048, 4096), "up_proj": (2048, 4096), "down_proj": (2048, 2048)},
            "t4_admission: routed columns disagree",
        ),
    ],
)
def test_routed_agreement_refuses_split(shapes, match):
    with pytest.raises(ValueError, match=match):
        require_routed_shape_agreement(shapes)


@pytest.mark.parametrize(
    "shapes,match",
    [
        (None, "t4_admission: routed scope needs"),
        ({"gate_proj": (2048, 4096)}, "t4_admission: routed scope lacks"),
        (
            {"gate_proj": (2048.0, 4096), "up_proj": (2048, 4096), "down_proj": (4096, 2048)},
            "t4_admission: gate_proj shape must be two integers",
        ),
        (
            {"gate_proj": (2048,), "up_proj": (2048, 4096), "down_proj": (4096, 2048)},
            "t4_admission: gate_proj shape must be two integers",
        ),
    ],
)
def test_routed_agreement_refuses_shape(shapes, match):
    with pytest.raises(ValueError, match=match):
        require_routed_shape_agreement(shapes)

