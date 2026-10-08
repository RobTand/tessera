"""Required class coordinates at the producer and serving boundary."""
import pytest

from tessera.serving.scheme import validate_tessera_moe_scheme


def _scheme():
    return {
        "family": "TESSERA_FP8", "structure": "routed_moe", "grid": "E4M3",
        "body": "WINDOW", "plane": "CHANNEL", "experts": 2,
        "groups": {
            "w13": {"q256": 1024, "rows": 64, "columns": 32,
                    "roles": [["gate_proj", 32], ["up_proj", 32]], "wire_stride": 512},
            "w2": {"q256": 1024, "rows": 32, "columns": 32,
                   "roles": [["down_proj", 32]], "wire_stride": 512},
        },
    }


@pytest.mark.parametrize("field", ["expert_ids", "expert_classes"])
def test_required_metadata_is_not_inferred(field):
    scheme = _scheme()
    scheme["expert_ids"] = [0, 1]
    scheme["expert_classes"] = [{"start": 0, "end": 2,
                                  "q256": {"w13": [1024, 1024], "w2": [1024]}}]
    del scheme[field]
    with pytest.raises(ValueError, match=field):
        validate_tessera_moe_scheme(scheme, "required metadata")


def test_producer_sorts_storage_without_renaming_sources():
    pytest.importorskip("torch")
    from tessera.export_serving import project_expert_plan

    stack = "model.layers.1.mlp.experts"
    shapes = {f"{stack}.{e}.{p}.weight": ([128, 512] if p == "down_proj" else [512, 128])
              for e in range(2) for p in ("gate_proj", "up_proj", "down_proj")}
    overrides = {f"{stack}.0.{p}": 1088 for p in ("gate_proj", "up_proj", "down_proj")}
    result = project_expert_plan(
        shapes, {"hidden_size": 128, "moe_intermediate_size": 512, "n_routed_experts": 2},
        {stack: {"grid": "E4M3", "q256": 1024, "unit_q256": overrides}})["stacks"][stack]
    assert result["expert_ids"] == [1, 0]
    assert [unit["expert"] for unit in result["units"]] == [0, 0, 0, 1, 1, 1]
    for unit in result["units"]:
        assert unit["source_tensor"] == unit["tensor"]
        assert unit["source_slice"]["expert"] == unit["expert"]
        assert unit["storage_expert"] == 1 - unit["expert"]
        assert unit["wire"] == f"{stack}.{unit['storage_expert']}.{unit['projection']}.wire"



def _metadata_scheme():
    scheme = _scheme()
    scheme["expert_ids"] = [0, 1]
    scheme["expert_classes"] = [{"start": 0, "end": 2,
                                  "q256": {"w13": [1024, 1024], "w2": [1024]}}]
    return scheme


@pytest.mark.parametrize("ids", [[], [0], [0, 0], [0, 2], [-1, 1], [False, 1], [0.0, 1], [1, 0]])
def test_malformed_bijections_and_within_class_order_refuse(ids):
    scheme = _metadata_scheme()
    scheme["expert_ids"] = ids
    with pytest.raises(ValueError):
        validate_tessera_moe_scheme(scheme, "bad map")


@pytest.mark.parametrize("classes", [
    [], [{"start": 0, "end": 0, "q256": {"w13": [1024, 1024], "w2": [1024]}}],
    [{"start": 1, "end": 2, "q256": {"w13": [1024, 1024], "w2": [1024]}}],
    [{"start": 0, "end": 1, "q256": {"w13": [1024, 1024], "w2": [1024]}}],
    [{"start": False, "end": 2, "q256": {"w13": [1024, 1024], "w2": [1024]}}],
    [{"start": 0, "end": 2, "q256": {"w13": [True, 1024], "w2": [1024]}}],
    [{"start": 0, "end": 2, "q256": {"w13": [1024], "w2": [1024]}}],
    [{"start": 0, "end": 2, "q256": {"w13": [1088, 1088], "w2": [1024]}}],
    [{"start": 0, "end": 2, "q256": {"w13": [1024, 1024], "w2": [1024]}},
     {"start": 1, "end": 2, "q256": {"w13": [1088, 1088], "w2": [1088]}}],
])
def test_malformed_partitions_and_class_rungs_refuse(classes):
    scheme = _metadata_scheme()
    scheme["expert_classes"] = classes
    with pytest.raises(ValueError):
        validate_tessera_moe_scheme(scheme, "bad classes")


@pytest.mark.parametrize("profiles", [
    [[1024, 1024, 1024], [1024, 1024, 1024]],
    [[1088, 1088, 1024], [1024, 1024, 1088], [1024, 1024, 1024]],
    [[1024, 1024, 1088], [1024, 1024, 1024], [1024, 1024, 1088], [1088, 1088, 1024]],
])
def test_full_profiles_sort_then_original_id_and_roundtrip(profiles):
    from tessera.expert_classes import build_expert_metadata, inverse_expert_ids, normalize_expert_metadata

    matrices = {"w13": [p[:2] for p in profiles], "w2": [p[2:] for p in profiles]}
    metadata = build_expert_metadata(matrices)
    expected = sorted(range(len(profiles)), key=lambda e: (profiles[e], e))
    assert metadata["expert_ids"] == expected
    inverse = inverse_expert_ids(expected)
    assert [inverse[e] for e in expected] == list(range(len(profiles)))
    stored = {group: [rows[e] for e in expected] for group, rows in matrices.items()}
    assert normalize_expert_metadata(**metadata, group_q256=stored) == metadata
    if all(p == profiles[0] for p in profiles):
        assert expected == list(range(len(profiles)))
        assert len(metadata["expert_classes"]) == 1


@pytest.mark.parametrize("profiles", [[1088, 1024], [1024, 1024]])
def test_duplicate_or_reversed_class_profiles_refuse(profiles):
    from tessera.expert_classes import normalize_expert_classes

    classes = [{"start": e, "end": e + 1, "q256": {"w13": [q, q], "w2": [q]}}
               for e, q in enumerate(profiles)]
    with pytest.raises(ValueError, match="strictly ordered"):
        normalize_expert_classes(classes, 2)


def test_gate_up_schedule_refuses_but_down_can_differ():
    scheme = _metadata_scheme()
    scheme["groups"]["w13"]["q256"] = [1024, 1088]
    scheme["expert_classes"][0]["q256"]["w13"] = [1024, 1088]
    with pytest.raises(ValueError, match="unservable_gate_up_schedule"):
        validate_tessera_moe_scheme(scheme, "gate/up")
    scheme["groups"]["w13"]["q256"] = 1024
    scheme["expert_classes"][0]["q256"]["w13"] = [1024, 1024]
    scheme["groups"]["w2"]["q256"] = 1088
    scheme["expert_classes"][0]["q256"]["w2"] = [1088]
    assert validate_tessera_moe_scheme(scheme, "down")["expert_classes"] == scheme["expert_classes"]


def test_original_plan_reconciliation_and_storage_mismatch():
    pytest.importorskip("torch")
    from copy import deepcopy
    from tessera.export_serving import project_expert_plan, expert_group_q256
    from tessera.serving_parts import validate_explicit_plan

    stack = "model.layers.1.mlp.experts"
    projections = ("gate_proj", "up_proj", "down_proj")
    shapes = {f"{stack}.{e}.{p}.weight": ([128, 512] if p == "down_proj" else [512, 128])
              for e in range(2) for p in projections}
    overrides = {f"{stack}.0.{p}": 1088 for p in projections}
    plan = {stack: {"grid": "E4M3", "q256": 1024, "unit_q256": overrides}}
    projected = project_expert_plan(shapes, {"hidden_size": 128, "moe_intermediate_size": 512,
                                           "n_routed_experts": 2}, plan)["stacks"][stack]
    scheme = _scheme()
    scheme.update({field: projected[field] for field in ("expert_ids", "expert_classes")})
    for group in scheme["groups"]:
        scheme["groups"][group].update({field: projected["groups"][group][field]
                                         for field in ("rows", "columns", "roles")})
        scheme["groups"][group]["q256"] = expert_group_q256(projected, group)
    roles = [dict(unit, role=unit["projection"], grid="E4M3",
                  q256=overrides.get(unit["tensor"].removesuffix(".weight"), 1024))
             for unit in projected["units"]]
    record = dict(projected, roles=roles, structure="routed_moe")
    groups = {"stack": {"targets": [stack], "scheme": scheme}}
    validate_explicit_plan(plan, {stack: record}, groups, source_tensors=set(shapes))
    bad_plan = deepcopy(plan)
    bad_plan[stack]["unit_q256"][f"{stack}.0.down_proj"] = 1024
    with pytest.raises(ValueError, match="rung differs from plan"):
        validate_explicit_plan(bad_plan, {stack: record}, groups)
    for field, value, error in [("storage_expert", 0, "storage_expert"),
                                ("wire", roles[0]["tensor"].replace(".weight", ".wire"), "wire"),
                                ("tensor", roles[0]["wire"].replace(".wire", ".weight"), "unknown projected")]:
        bad_record = deepcopy(record)
        bad_record["roles"][0][field] = value
        with pytest.raises(ValueError, match=error):
            validate_explicit_plan(plan, {stack: bad_record}, groups)

