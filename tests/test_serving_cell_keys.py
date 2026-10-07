"""Kernel build keys preserve module and execution scope."""
import copy
import pytest
from tessera.serving import contract
from tessera.serving.census import cell_launch_agreement


def _cell():
    return {"id": "old_cell", "platform": "sm_121", "family": "TESSERA_E4M3_K1",
        "structure": "dense", "regime": "decode", "rungs_q256": [1024],
        "runtime": {"image": "example/runtime@sha256:" + "a" * 64,
                    "execution_modes": ["eager"], "kernel_build": "build-a"},
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
        "executes": [{"symbol": "native.kernel", "decoder": "native_decoder"}]}


def test_key_does_not_depend_on_image_digest():
    cell = _cell()
    other = copy.deepcopy(cell)
    other["runtime"]["image"] = "example/other@sha256:" + "b" * 64
    assert contract.cell_kernel_key(cell) == contract.cell_kernel_key(other)
    other["structure"] = "routed_moe"
    assert contract.cell_kernel_key(cell) != contract.cell_kernel_key(other)
    other = copy.deepcopy(cell)
    other["runtime"]["kernel_build"] = "build-b"
    assert contract.cell_kernel_key(cell) != contract.cell_kernel_key(other)


def test_compatibility_map_preserves_old_ids():
    cells = [_cell()]
    assert contract.cell_key_compatibility(cells)["old_cell"] == contract.cell_kernel_key(cells[0])


def test_census_matches_build_across_images_but_checks_launch():
    records = {"decode": {"model.proj": {"kind": "dense", "policy": "TESSERA_FP8:resident",
        "symbol": "native.kernel", "decoder": "native_decoder", "shape": "M1:N128:K128"}}}
    options = dict(cells=[_cell()], phase_regimes={"decode": "decode"}, platform="sm_121",
        structure="dense", rungs_by_module={"model.proj": 1024},
        families_by_route={"TESSERA_FP8": "TESSERA_E4M3_K1"},
        runtime_image="example/new@sha256:" + "b" * 64, execution_mode="eager", kernel_build="build-a")
    block, problems = cell_launch_agreement(records, **options)
    assert block["agrees"] is True and not problems
    records["decode"]["model.proj"]["symbol"] = "wrong.kernel"
    block, problems = cell_launch_agreement(records, **options)
    assert block["agrees"] is False and problems
    options["kernel_build"] = "build-b"
    block, problems = cell_launch_agreement(records, **options)
    assert block["agrees"] is None and not problems


@pytest.mark.parametrize("value", ["", None, 1, " build-a", "build-a\n"])
def test_kernel_build_has_a_complete_name(value):
    cell = _cell()
    cell["runtime"]["kernel_build"] = value
    with pytest.raises(ValueError, match="kernel_build"):
        contract.cell_kernel_key(cell)


def test_published_keys_preserve_receipts_and_measurement_scope():
    import json
    from pathlib import Path
    import subprocess
    root = Path(__file__).resolve().parents[1]
    old = json.loads(subprocess.check_output(["git", "show",
        "3fa776859b7dda9b1e4003c6ddd988e30b2b85db:src/tessera/serving/runtime_contract.json"], cwd=root))
    current = contract.load_serving_contract()
    mapping = contract.cell_key_compatibility(current["lane_eligibility"]["cells"])
    assert set(mapping) == {cell["id"] for cell in old["lane_eligibility"]["cells"]}
    stripped = copy.deepcopy(current["lane_eligibility"]["cells"])
    for cell in stripped:
        assert cell["runtime"].pop("kernel_build").startswith("legacy-toolchain/")
    assert stripped == old["lane_eligibility"]["cells"]
    for field in ("versions", "formats", "tensor_parallel", "expert_parallel", "native_extensions"):
        assert current[field] == old[field]


def test_same_kernel_scope_cannot_hide_behind_another_image():
    payload = contract.load_serving_contract()
    cell = copy.deepcopy(payload["lane_eligibility"]["cells"][0])
    cell["runtime"]["image"] = "example/other@sha256:" + "c" * 64
    cell["id"] += contract.cell_runtime_id_suffix(cell)
    payload["lane_eligibility"]["cells"].append(cell)
    with pytest.raises(ValueError, match="both cover"):
        contract.validate_serving_contract(payload)


def test_image_only_scope_cannot_overlap_across_build_names():
    payload = contract.load_serving_contract()
    cell = copy.deepcopy(payload["lane_eligibility"]["cells"][0])
    cell["runtime"]["kernel_build"] = "build-b"
    cell["id"] += contract.cell_runtime_id_suffix(cell)
    payload["lane_eligibility"]["cells"].append(cell)
    with pytest.raises(ValueError, match="both cover"):
        contract.validate_serving_contract(payload)
