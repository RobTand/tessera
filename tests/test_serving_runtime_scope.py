"""Attested route cells are bounded by the image and execution they measured."""
from __future__ import annotations

import copy

import pytest

from tessera.serving import contract as runtime_contract


IMAGE = "example/runtime@sha256:" + "a" * 64
OTHER_IMAGE = "example/runtime@sha256:" + "b" * 64


def _scoped_contract():
    contract = runtime_contract.load_serving_contract()
    for cell in contract["lane_eligibility"]["cells"]:
        cell["runtime"] = {"image": IMAGE, "execution_modes": ["eager", "compiled"]}
    return contract


def test_every_published_cell_names_its_measured_runtime():
    contract = runtime_contract.load_serving_contract()
    assert contract["lane_eligibility"]["schema"] == "tessera.lane-eligibility.v5"
    for cell in contract["lane_eligibility"]["cells"]:
        image, modes = runtime_contract.cell_runtime_scope(cell)
        assert image and modes
        if cell["structure"] == "dense":
            assert image == contract["versions"]["attested_on"]["image"]
            assert modes == ("eager", "compiled")


@pytest.mark.parametrize("image", [None, 1, "", "example/runtime:latest",
                                  "sha256:" + "a" * 64,
                                  "example/runtime@sha256:" + "A" * 64,
                                  "example/runtime@sha256:" + "a" * 63,
                                  IMAGE + "\n"])
def test_runtime_image_requires_an_exact_manifest_digest(image):
    with pytest.raises(ValueError, match="digest reference"):
        runtime_contract.require_runtime_image(image)


def test_cell_runtime_scope_uses_explicit_image_and_canonical_mode_order():
    cell = {"runtime": {"image": IMAGE, "execution_modes": ["compiled", "eager"]}}
    assert runtime_contract.cell_runtime_scope(cell) == (IMAGE, ("eager", "compiled"))
    assert runtime_contract.require_runtime_image(IMAGE) == IMAGE


@pytest.mark.parametrize("modes", [None, [], "eager", ["eager", "eager"],
                                  ["unknown"], [1], ["eager", None]])
def test_cell_runtime_execution_modes_are_explicit_nonempty_and_unique(modes):
    cell = {"runtime": {"image": IMAGE, "execution_modes": modes}}
    with pytest.raises(ValueError, match="execution_modes"):
        runtime_contract.cell_runtime_scope(cell)


@pytest.mark.parametrize("scope", [None, {}, {"image": IMAGE},
                                  {"image": IMAGE, "execution_modes": ["eager"],
                                   "rationale": "trust this runtime"}])
def test_cell_runtime_scope_is_a_closed_object(scope):
    with pytest.raises(ValueError, match="runtime"):
        runtime_contract.cell_runtime_scope({"runtime": scope})


def test_a_cell_may_not_borrow_the_global_dense_image_pin():
    contract = runtime_contract.load_serving_contract()
    for cell in contract["lane_eligibility"]["cells"]:
        cell.pop("runtime", None)
    with pytest.raises(ValueError, match="runtime"):
        runtime_contract.validate_serving_contract(contract)


def test_runtime_variants_have_disjoint_scopes_and_unique_ids():
    contract = _scoped_contract()
    first = contract["lane_eligibility"]["cells"][0]
    first["runtime"]["execution_modes"] = ["eager"]
    other_mode = copy.deepcopy(first)
    other_mode["runtime"]["execution_modes"] = ["compiled"]
    other_mode["id"] += runtime_contract.cell_runtime_id_suffix(other_mode)
    other_image = copy.deepcopy(first)
    other_image["runtime"]["image"] = OTHER_IMAGE
    other_image["id"] += runtime_contract.cell_runtime_id_suffix(other_image)
    contract["lane_eligibility"]["cells"].extend([other_mode, other_image])
    runtime_contract.validate_serving_contract(contract)

    # Matching explicit fields, not the label, decides whether two cells
    # collide. A runtime-specific id cannot conceal an overlapping scope.
    other_mode["runtime"]["execution_modes"] = ["eager", "compiled"]
    other_mode["id"] = first["id"] + runtime_contract.cell_runtime_id_suffix(other_mode)
    with pytest.raises(ValueError, match="both cover"):
        runtime_contract.validate_serving_contract(contract)


def test_different_runtime_scopes_cannot_reuse_one_cell_id():
    contract = _scoped_contract()
    variant = copy.deepcopy(contract["lane_eligibility"]["cells"][0])
    variant["runtime"]["image"] = OTHER_IMAGE
    contract["lane_eligibility"]["cells"].append(variant)
    with pytest.raises(ValueError, match="repeats.*id"):
        runtime_contract.validate_serving_contract(contract)


def test_runtime_id_suffix_is_derived_from_the_entire_scope():
    cell = {"runtime": {"image": IMAGE, "execution_modes": ["eager", "compiled"]}}
    reverse = {"runtime": {"image": IMAGE, "execution_modes": ["compiled", "eager"]}}
    suffix = runtime_contract.cell_runtime_id_suffix(cell)
    assert suffix == runtime_contract.cell_runtime_id_suffix(reverse)
    for variant in ({"image": OTHER_IMAGE, "execution_modes": ["eager", "compiled"]},
                    {"image": IMAGE, "execution_modes": ["eager"]}):
        assert suffix != runtime_contract.cell_runtime_id_suffix({"runtime": variant})


def test_validator_reads_runtime_scope_instead_of_accepting_prose():
    contract = _scoped_contract()
    contract["lane_eligibility"]["cells"][0]["runtime"]["image"] = "example/runtime:latest"
    with pytest.raises(ValueError, match="digest reference"):
        runtime_contract.validate_serving_contract(contract)
