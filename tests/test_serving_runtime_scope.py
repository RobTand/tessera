"""Attested route cells are bounded by the image, execution and toolchain they measured.

Schema v5 gave every cell its ``runtime`` scope -- the exact image and the
execution modes a receipt covers.  Schema v6 (#131) adds the TOOLCHAIN the
cell's own receipt records, ``vllm`` and ``torch``, because until then the
only place a version lived was the global ``versions.attested_on`` block,
which named vLLM 0.28.0 for cells measured on ``0.28.1rc1.dev397``.  Two
helpers, one home for the key sets: :func:`cell_runtime_scope` reads the
scope (the census's join key; the version keys are optional there so an
observation-side fixture that carries only a scope keeps working) and
:func:`cell_runtime_versions` requires the whole closed object, which is
what the validator calls.
"""
from __future__ import annotations

import copy

import pytest

from tessera.serving import contract as runtime_contract


IMAGE = "example/runtime@sha256:" + "a" * 64
OTHER_IMAGE = "example/runtime@sha256:" + "b" * 64
RUNTIME = {"image": IMAGE, "execution_modes": ["eager", "compiled"],
           "vllm": "0.99.0", "torch": "2.99.0+cu999"}


def _scoped_contract():
    contract = runtime_contract.load_serving_contract()
    for cell in contract["lane_eligibility"]["cells"]:
        cell["runtime"] = copy.deepcopy(RUNTIME)
    contract["versions"]["default_serve_image"] = IMAGE
    return contract


def test_every_published_cell_names_its_measured_runtime_and_toolchain():
    contract = runtime_contract.load_serving_contract()
    # The constant, not a literal: this test is about what a cell must carry,
    # and the schema string it carries it under is the packaged code's to say
    # (AGENTS.md rule 3).  Written as "...v6" here, it failed on the v7 bump
    # while every claim it actually makes still held.
    assert (contract["lane_eligibility"]["schema"]
            == runtime_contract.LANE_ELIGIBILITY_SCHEMA)
    for cell in contract["lane_eligibility"]["cells"]:
        image, modes = runtime_contract.cell_runtime_scope(cell)
        vllm, torch = runtime_contract.cell_runtime_versions(cell)
        assert image and modes and vllm and torch
        assert set(cell["runtime"]) == (runtime_contract.RUNTIME_SCOPE_KEYS
                                        | runtime_contract.RUNTIME_VERSION_KEYS)


def test_one_image_carries_one_toolchain():
    """Two cells naming one image and two vLLM builds would be two runtimes
    under one digest, which a digest cannot be."""
    contract = runtime_contract.load_serving_contract()
    by_image = {}
    for cell in contract["lane_eligibility"]["cells"]:
        by_image.setdefault(cell["runtime"]["image"], set()).add(
            runtime_contract.cell_runtime_versions(cell))
    assert all(len(versions) == 1 for versions in by_image.values()), by_image
    assert len(by_image) == 2, "the dense pin and the EUGR MoE image, nothing else"


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


def test_cell_runtime_versions_requires_the_whole_closed_object():
    assert runtime_contract.cell_runtime_versions({"runtime": RUNTIME}) == ("0.99.0", "2.99.0+cu999")
    with pytest.raises(ValueError, match=r"runtime is missing \['torch', 'vllm'\]"):
        runtime_contract.cell_runtime_versions(
            {"runtime": {"image": IMAGE, "execution_modes": ["eager"]}})
    with pytest.raises(ValueError, match=r"runtime is missing \['torch'\]"):
        runtime_contract.cell_runtime_versions(
            {"runtime": {"image": IMAGE, "execution_modes": ["eager"], "vllm": "0.99.0"}})
    with pytest.raises(ValueError, match=r"unknown field\(s\) \['device'\]"):
        runtime_contract.cell_runtime_versions(
            {"runtime": {**RUNTIME, "device": "NVIDIA GB10 (sm_121)"}})
    for field in ("vllm", "torch"):
        for bad in ("", None, 28):
            with pytest.raises(ValueError, match=f"runtime.{field} must be a non-empty string"):
                runtime_contract.cell_runtime_versions({"runtime": {**RUNTIME, field: bad}})


def test_the_validator_requires_the_toolchain_on_every_cell():
    contract = _scoped_contract()
    del contract["lane_eligibility"]["cells"][0]["runtime"]["vllm"]
    with pytest.raises(ValueError, match=r"runtime is missing \['vllm'\]"):
        runtime_contract.validate_serving_contract(contract)


def test_the_validator_refuses_two_toolchains_under_one_image():
    contract = _scoped_contract()
    contract["lane_eligibility"]["cells"][1]["runtime"]["vllm"] = "0.99.1"
    with pytest.raises(ValueError, match="one image cannot be two runtimes"):
        runtime_contract.validate_serving_contract(contract)


def test_a_cell_may_not_borrow_the_global_default_image_pin():
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


def test_runtime_id_suffix_is_derived_from_the_scope_not_the_toolchain():
    """The suffix distinguishes disjoint SCOPES; a toolchain is a property of
    the image, so it must not fork the id (two ids for one scope would be the
    table-order defect the suffix exists to close)."""
    cell = {"runtime": {"image": IMAGE, "execution_modes": ["eager", "compiled"]}}
    reverse = {"runtime": {"image": IMAGE, "execution_modes": ["compiled", "eager"]}}
    suffix = runtime_contract.cell_runtime_id_suffix(cell)
    assert suffix == runtime_contract.cell_runtime_id_suffix(reverse)
    assert suffix == runtime_contract.cell_runtime_id_suffix({"runtime": RUNTIME})
    for variant in ({"image": OTHER_IMAGE, "execution_modes": ["eager", "compiled"]},
                    {"image": IMAGE, "execution_modes": ["eager"]}):
        assert suffix != runtime_contract.cell_runtime_id_suffix({"runtime": variant})


def test_validator_reads_runtime_scope_instead_of_accepting_prose():
    contract = _scoped_contract()
    contract["lane_eligibility"]["cells"][0]["runtime"]["image"] = "example/runtime:latest"
    with pytest.raises(ValueError, match="digest reference"):
        runtime_contract.validate_serving_contract(contract)
