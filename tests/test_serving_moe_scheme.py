"""The routed-MoE sidecar scheme: two groups, E experts, one route.

A dense scheme describes one module -- one container, one exact byte count.
An expert stack holds gate, up and down containers for each expert, whose
lengths differ by projection and expert
(``tessera.moe_layout`` says why: the manifest writes ``global_scale`` as an
exact varint ratio, so the blob length follows the data even at fixed shape
and rung).  So the MoE scheme declares the expert count, the two groups vLLM's
fused-MoE kernel reads (``w13`` = gate then up, ``w2`` = down), and per group
a ``wire_stride`` -- the parameter row width every expert's blob is copied
into -- instead of a ``wire_bytes``.

What these tests pin is that the two shapes share ONE geometry check
(``_validate_group``), that the module-level facts a route is selected by
cannot be restated per group, and that the two groups' geometries are checked
against each other rather than each against nothing.  ``STRUCTURES`` is a
separate question and has its own test here: what this build PARSES is wider
than what it DISPATCHES, and the parser is not a promise to serve.
"""
from __future__ import annotations

import pytest

from tessera.serving import scheme as S


def _group(rows, columns, roles, q256=1024, stride=4096, **over):
    g = {"rows": rows, "columns": columns, "roles": roles, "q256": q256, "wire_stride": stride}
    g.update(over)
    return g


def _moe(experts=4, hidden=128, inter=64, **over):
    s = {
        "family": S.TESSERA_FP8, "structure": S.STRUCTURE_ROUTED_MOE,
        "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL",
        "experts": experts,
        "groups": {
            "w13": _group(2 * inter, hidden, [["gate_proj", inter], ["up_proj", inter]]),
            "w2": _group(hidden, inter, [["down_proj", hidden]]),
        },
    }
    s.update(over)
    return s


def test_a_moe_scheme_normalises_to_groups_and_the_two_sizes():
    norm = S.validate_tessera_moe_scheme(_moe(), "m")
    assert norm["structure"] == S.STRUCTURE_ROUTED_MOE and norm["experts"] == 4
    assert norm["hidden_size"] == 128 and norm["intermediate_size"] == 64
    assert sorted(norm["groups"]) == ["w13", "w2"]
    w13 = norm["groups"]["w13"]
    assert w13["roles"] == [("gate_proj", 64), ("up_proj", 64)]
    assert w13["role_q256"] == [1024, 1024] and w13["wire_stride"] == 4096
    assert norm["groups"]["w2"]["roles"] == [("down_proj", 128)]


def test_the_source_layout_stamp_is_closed_and_old_wires_default_to_unpacked():
    for layout in S.MOE_SOURCE_LAYOUTS:
        assert S.validate_tessera_moe_scheme(
            _moe(source_layout=layout), "m")["source_layout"] == layout
    assert S.validate_tessera_moe_scheme(
        _moe(), "old")["source_layout"] == S.MOE_SOURCE_UNPACKED
    with pytest.raises(ValueError, match="source_layout"):
        S.validate_tessera_moe_scheme(_moe(source_layout="guessed"), "m")


def test_the_group_check_is_the_dense_check_so_a_route_fact_refuses_the_same_way():
    with pytest.raises(ValueError, match="scalar E4M3 grid"):
        S.validate_tessera_moe_scheme(_moe(grid="E2M1x2"), "m")
    with pytest.raises(ValueError, match="no FP8 tile"):
        S.validate_tessera_moe_scheme(_moe(plane="LUT"), "m")
    with pytest.raises(ValueError, match="K % 16"):
        S.validate_tessera_moe_scheme(_moe(hidden=120), "m")
    with pytest.raises(ValueError, match="roles stack to"):
        bad = _moe()
        bad["groups"]["w2"]["roles"] = [["down_proj", 64]]
        S.validate_tessera_moe_scheme(bad, "m")


def test_a_group_may_not_restate_a_module_fact_differently():
    """vLLM builds ONE method per expert stack, so grid/body/plane/family are
    module facts.  A group that says otherwise is two answers about one tile."""
    bad = _moe()
    bad["groups"]["w2"]["grid"] = "E2M1x2"
    with pytest.raises(ValueError, match="disagrees with the module"):
        S.validate_tessera_moe_scheme(bad, "m")


def test_the_group_set_is_exactly_the_tiles_the_kernel_reads():
    bad = _moe()
    bad["groups"]["w3"] = bad["groups"]["w2"]
    with pytest.raises(ValueError, match="exactly the groups"):
        S.validate_tessera_moe_scheme(bad, "m")
    missing = _moe()
    del missing["groups"]["w2"]
    with pytest.raises(ValueError, match="exactly the groups"):
        S.validate_tessera_moe_scheme(missing, "m")


def test_a_group_holds_exactly_the_shards_the_runtime_loads_into_it():
    bad = _moe()
    bad["groups"]["w13"]["roles"] = [["gate_proj", 128]]
    with pytest.raises(ValueError, match="expected 2"):
        S.validate_tessera_moe_scheme(bad, "m")


@pytest.mark.parametrize("group,roles", [
    ("w13", [["up_proj", 64], ["gate_proj", 64]]),
    ("w13", [["gate_proj", 64], ["gate_proj", 64]]),
    ("w13", [["w1", 64], ["w3", 64]]),
    ("w2", [["gate_proj", 128]]),
])
def test_group_roles_must_name_the_runtime_projections_in_row_order(group, roles):
    # A matching sidecar and blob could otherwise agree on the wrong role:
    # the runtime still treats the first w13 half as gate and the second as up.
    bad = _moe()
    bad["groups"][group]["roles"] = roles
    with pytest.raises(ValueError, match=rf"group '{group}': roles.*row order"):
        S.validate_tessera_moe_scheme(bad, "m")


@pytest.mark.parametrize("gate_rows,up_rows", [(32, 96), (63, 65)])
def test_gate_and_up_boundaries_must_match_the_runtime_equal_halves(gate_rows, up_rows):
    bad = _moe()
    bad["groups"]["w13"]["roles"] = [["gate_proj", gate_rows], ["up_proj", up_rows]]
    # Total rows and the cross-group geometry still agree. Only the role
    # boundary is wrong: vLLM applies gate/up at N, not at the declared split.
    with pytest.raises(ValueError, match="w13.*role rows.*equal halves"):
        S.validate_tessera_moe_scheme(bad, "m")


def test_the_two_groups_geometries_are_checked_against_each_other():
    """w13 is [2N, K] and w2 is [K, N] over ONE expert: two statements of two
    numbers, so a checkpoint whose halves disagree is caught before a decode
    lands 2N rows in a K-row tile."""
    bad = _moe()
    bad["groups"]["w2"]["columns"] = 32
    bad["groups"]["w2"]["rows"] = 128
    with pytest.raises(ValueError, match="twice w2's columns"):
        S.validate_tessera_moe_scheme(bad, "m")
    other = _moe()
    other["groups"]["w2"]["rows"] = 256
    other["groups"]["w2"]["roles"] = [["down_proj", 256]]
    with pytest.raises(ValueError, match="both are the model's"):
        S.validate_tessera_moe_scheme(other, "m")


def test_the_expert_count_is_a_positive_integer():
    with pytest.raises(ValueError, match="experts must be positive"):
        S.validate_tessera_moe_scheme(_moe(experts=0), "m")
    with pytest.raises(ValueError, match="experts must be an integer"):
        S.validate_tessera_moe_scheme(_moe(experts="4"), "m")


def test_dispatch_is_narrower_than_parse_and_says_so():
    """The parser reads a routed_moe scheme; ``STRUCTURES`` says whether this
    build serves one.  Two questions, and the gate that answers the second is
    the one ``config.get_quant_method`` reads."""
    if S.STRUCTURE_ROUTED_MOE in S.STRUCTURES:
        assert S.validate_tessera_scheme(_moe(), "m")["structure"] == S.STRUCTURE_ROUTED_MOE
    else:
        with pytest.raises(ValueError, match="is not served"):
            S.validate_tessera_scheme(_moe(), "m")


def test_the_dense_shape_is_untouched_by_the_moe_shape():
    dense = {"family": S.TESSERA_FP8, "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL",
             "q256": 1024, "rows": 256, "columns": 1024, "wire_bytes": 4096,
             "roles": [["weight", 256]]}
    norm = S.validate_tessera_scheme(dense, "d")
    assert norm["structure"] == S.STRUCTURE_DENSE and norm["wire_bytes"] == 4096
    assert norm["roles"] == [("weight", 256)] and norm["q256"] == 1024
    with pytest.raises(ValueError, match="missing"):
        S.validate_tessera_scheme({k: v for k, v in dense.items() if k != "roles"}, "d")
