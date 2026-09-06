"""Mixed census coverage counts fused dense owners and expert projections separately."""
import copy

import pytest

from tessera.serving.contract import construction_entry, output_partitions
from tessera.serving.scheme import ROUTES, TESSERA_BF16, TESSERA_NVFP4, launch_pairs
from test_ts5_census_check import IMAGE, TARGETS, _check, _fixture, _promote


def _add_dense(case, target, members, *, family=TESSERA_BF16, grid="BF16", q256=1792):
    plan, config, manifest, census, contract = case
    route = ROUTES[family]
    # Rows are whatever the pinned LFM census attests for this owner's output
    # partitions (tessera#377); a member with no attested geometry keeps 64.
    entry = construction_entry(config["architectures"], contract)
    sizes = output_partitions(entry, target) if entry is not None else None
    if sizes is None or len(sizes) != len(members):
        sizes = [64] * len(members)
    roles = [{"tensor": name, "role": name.removesuffix(".weight").rsplit(".", 1)[-1],
              "rows": rows, "cols": 128, "grid": grid, "q256": q256, "family": family}
             for name, rows in zip(members, sizes)]
    for name in members:
        plan[name] = {"grid": grid, "q256": q256}
    scheme = {"structure": "dense", "family": family, "grid": grid,
              "body": route["body"], "plane": route["plane"], "q256": q256,
              "rows": sum(r["rows"] for r in roles), "columns": 128, "wire_bytes": 4096,
              "roles": [[r["role"], r["rows"]] for r in roles]}
    config["quantization_config"]["config_groups"][target] = {
        "targets": [target], "format": "TESSERA", "scheme": scheme}
    manifest["modules"][target] = {"family": family, "grid": grid, "q256": q256,
        "rows": scheme["rows"], "cols": 128, "container_bytes": 4096, "roles": roles}
    manifest["export_identity"]["options"]["plan"] = copy.deepcopy(plan)
    manifest["totals"]["modules"] += 1
    manifest["totals"]["units"] += len(roles)
    for phase, regime in (("decode", "decode"), ("prefill", "batch")):
        symbol, decoder = sorted(launch_pairs(family, structure="dense", regime=regime,
                                             mode="resident"))[0]
        census["records"][phase][target] = {"kind": "dense", "state": "served",
            "policy": family + ":resident", "contract": route["activation_contract"],
            "symbol": symbol, "decoder": decoder,
            "shape": f"M{1 if phase == 'decode' else 64}:N{scheme['rows']}:K128"}
        census["record_owner"][phase][target] = target
    census["declared_name_mapping"][target] = target


DENSE = "model.layers.0.feed_forward.w13"
MEMBERS = ["model.layers.0.feed_forward.w1.weight", "model.layers.0.feed_forward.w3.weight"]


def _mixed():
    case = _fixture()
    _add_dense(case, DENSE, MEMBERS, family=TESSERA_NVFP4, grid="E2M1x2", q256=896)
    _add_dense(case, "model.layers.0.self_attn.out_proj",
               ["model.layers.0.self_attn.out_proj.weight"])
    return case


def test_mixed_population_joins_dense_source_leaves_to_one_fused_owner():
    result = _check(_mixed())
    assert len(result["expected_owners"]) == 4
    assert result["expected_projection_units"] == 15
    assert set(result["cell_launch_agreement"]["structures"]) == {"dense", "routed_moe"}
    assert all(p["unattested"] == 2 for p in
               result["cell_launch_agreement"]["structures"]["dense"]["phases"].values())


def test_full_lfm_mix_counts_74_owners_and_2178_matrices():
    case = _fixture()
    plan, config, manifest, census, _ = case
    old_targets = list(TARGETS)
    # Eleven copies of the two-stack fixture, each with 32 experts: 22*32*3.
    new_groups, new_modules, new_plan, new_records, new_owners = {}, {}, {}, {}, {}
    for owner in range(22):
        old = old_targets[owner % 2]
        target = f"model.layers.{owner + 2}.feed_forward.experts"
        scheme = copy.deepcopy(config["quantization_config"]["config_groups"][old])
        scheme["targets"] = [target]
        scheme["scheme"]["experts"] = 32
        module = copy.deepcopy(manifest["modules"][old])
        base_roles = [r for r in module["roles"] if r["expert"] == 0]
        module["experts"] = 32
        module["roles"] = [{**r, "expert": expert} for expert in range(32) for r in base_roles]
        new_groups[target], new_modules[target] = scheme, module
        new_plan[target] = {"grid": "E4M3", "q256": 1024}
        for phase in ("decode", "prefill"):
            child = target + ".routed_experts"
            new_records.setdefault(phase, {})[child] = copy.deepcopy(census["records"][phase][old + ".routed_experts"])
            new_owners.setdefault(phase, {})[child] = target
    plan.clear()
    plan.update(new_plan)
    plan["lm_head.weight"] = "BF16"
    config["quantization_config"]["config_groups"] = new_groups
    manifest["modules"] = new_modules
    manifest["routed_moe"] = {"quantized_stacks": list(new_plan), "quantized_logical_units": 2112}
    manifest["totals"] = {"modules": 22, "units": 2112}
    census["records"], census["record_owner"] = new_records, new_owners
    census["declared_name_mapping"] = {t: t for t in new_plan}
    for layer in (0, 1):
        prefix = f"model.layers.{layer}.feed_forward."
        for owner, names in (("w13", ("w1", "w3")), ("w2", ("w2",))):
            _add_dense(case, prefix + owner, [prefix + n + ".weight" for n in names],
                       family=TESSERA_NVFP4, grid="E2M1x2", q256=896)
    for layer in range(48):
        prefix = f"model.layers.{layer}.self_attn."
        names = ("q_proj", "k_proj", "v_proj") if layer < 6 else ("out_proj",)
        owner = "qkv_proj" if layer < 6 else "out_proj"
        _add_dense(case, prefix + owner, [prefix + n + ".weight" for n in names])
    result = _check(case)
    assert len(result["expected_owners"]) == 74
    assert result["expected_projection_units"] == 2178


def test_routed_attestation_does_not_promote_unattested_dense_owners():
    case = _mixed()
    _promote(case)
    result = _check(case)
    assert result["cell_launch_agreement"]["structures"]["routed_moe"]["agrees"] is True
    with pytest.raises(ValueError, match="attest.*dense"):
        _check(case, require_attested=True)


@pytest.mark.parametrize("defect", ["shape", "role_order", "role_geometry", "role_missing",
    "role_extra", "unplanned", "passthrough", "scheme", "kind", "launch"])
def test_mixed_dense_obligations_fail_by_name(defect):
    case = _mixed()
    plan, config, manifest, census, _ = case
    module = manifest["modules"][DENSE]
    record = census["records"]["decode"][DENSE]
    if defect == "shape":
        record["shape"] = "M1:N64:K128"
    elif defect == "role_order":
        module["roles"].reverse()
    elif defect == "role_geometry":
        module["roles"][0]["cols"] = 64
    elif defect == "role_missing":
        module["roles"].pop()
    elif defect == "role_extra":
        module["roles"].append(copy.deepcopy(module["roles"][0]))
    elif defect == "unplanned":
        module["roles"][0]["tensor"] = "model.unplanned.weight"
    elif defect == "passthrough":
        plan[MEMBERS[0]] = "BF16"
        manifest["export_identity"]["options"]["plan"] = copy.deepcopy(plan)
    elif defect == "scheme":
        config["quantization_config"]["config_groups"][DENSE]["scheme"]["roles"].reverse()
    elif defect == "kind":
        record["kind"] = "moe"
    else:
        record["symbol"] = "wrong.kernel"
    with pytest.raises(ValueError, match="dense|explicit plan tensor|planned BF16"):
        _check(case)


def test_dense_source_roles_cannot_be_swapped_between_equal_shaped_owners():
    case = _mixed()
    other = "model.layers.1.feed_forward.w13"
    members = [name.replace("layers.0.", "layers.1.") for name in MEMBERS]
    _add_dense(case, other, members, family=TESSERA_NVFP4, grid="E2M1x2", q256=896)
    modules = case[2]["modules"]
    for left, right in zip(modules[DENSE]["roles"], modules[other]["roles"]):
        left["tensor"], right["tensor"] = right["tensor"], left["tensor"]
    with pytest.raises(ValueError, match="dense.*owner"):
        _check(case)


def test_dense_plan_cannot_split_one_fusion_into_two_census_owners():
    case = _fixture()
    for name in MEMBERS:
        _add_dense(case, name.removesuffix(".weight"), [name],
                   family=TESSERA_NVFP4, grid="E2M1x2", q256=896)
    with pytest.raises(ValueError, match="dense.*owner"):
        _check(case)
