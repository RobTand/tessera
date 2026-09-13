"""``versions`` says one thing per field, and every field is validated (#131).

Until lane-eligibility schema v6 ``versions.attested_on`` did double duty: it
was read as "the toolchain this contract was written against" AND as "the
runtime the cells were measured on", and it was factually wrong for the two
``routed_moe`` cells, which were measured on a different image and a
different vLLM build than the block named.  The block was required by the
validator and nothing inside it was checked.

v6 separates the two meanings.  The toolchain a cell was measured on lives on
the CELL (``runtime.vllm``, ``runtime.torch``, beside the image and modes it
already scoped -- ``tests/test_serving_runtime_scope.py``), read from each
cell's own receipt.  ``versions`` keeps only what is about the package:
``tessera`` (the distribution version), ``plugin_entry_point`` (the entry
point the wheel declares) and ``default_serve_image`` -- the ONE serve-image
pin every harness reads (``runtime_image.PIN_CONTRACT_FIELD``), which must be
an image some cell attests, because a default nobody measured is not a pin.
"""
from __future__ import annotations

import copy

import pytest

from tessera.serving.contract import (
    PLUGIN_ENTRY_POINT,
    VERSIONS_KEYS,
    load_serving_contract,
    require_runtime_image,
    validate_serving_contract,
)
from tessera.serving.runtime_image import PIN_CONTRACT_FIELD


@pytest.fixture(scope="module")
def contract():
    return load_serving_contract()


def _with_versions(contract, versions):
    doc = copy.deepcopy(contract)
    doc["versions"] = versions
    return doc


def test_versions_is_closed_and_says_nothing_about_a_measured_runtime(contract):
    assert VERSIONS_KEYS == frozenset({"tessera", "plugin_entry_point", "default_serve_image"})
    assert set(contract["versions"]) == VERSIONS_KEYS
    assert "attested_on" not in contract["versions"]


def test_the_pin_is_the_default_serve_image_and_is_an_attested_cell_image(contract):
    assert PIN_CONTRACT_FIELD == ("versions", "default_serve_image")
    pin = contract["versions"]["default_serve_image"]
    assert require_runtime_image(pin) == pin
    attested = {cell["runtime"]["image"] for cell in contract["lane_eligibility"]["cells"]}
    assert pin in attested


def test_the_default_serve_image_is_the_dense_cells_image(contract):
    """What the pin denotes did not move, said PER PLATFORM (schema v10).

    It was one sentence about the whole table, which worked while one
    platform had cells. With a platform axis the same claim has to name the
    platform it is about, or the day a second platform publishes cells the
    dense set is a union and the assertion means nothing.

    The claim, per platform that has cells: its dense cells were all measured
    on that platform's ``serve_image``, no ``routed_moe`` cell shares it, and
    the pin is some platform's serve image. Two images on one platform is the
    shipped fact this is written around -- ``sm_121``'s eight dense cells are
    on the vanilla pin and its two MoE cells on a second build -- which is why
    the validator's rule is "attested by one of its own cells" rather than
    "every cell's image".
    """
    block = contract["lane_eligibility"]
    pin = contract["versions"]["default_serve_image"]
    by_platform = {}
    for cell in block["cells"]:
        by_platform.setdefault(cell["platform"], {}).setdefault(
            cell["structure"], set()).add(cell["runtime"]["image"])
    assert by_platform, "no platform has cells; the premise moved"
    for platform, by_structure in by_platform.items():
        serve_image = block["platforms"][platform]["serve_image"]
        assert by_structure["dense"] == {serve_image}, platform
        assert serve_image not in by_structure.get("routed_moe", set()), platform
    assert pin in {block["platforms"][p]["serve_image"] for p in by_platform}
    for platform, entry in block["platforms"].items():
        if platform not in by_platform:
            assert entry["serve_image"] is None, (
                f"{platform} names a serve image with no cell to have measured it")


def test_the_entry_point_is_the_one_the_package_registers(contract):
    assert PLUGIN_ENTRY_POINT == "tessera = tessera.serving:register"
    assert contract["versions"]["plugin_entry_point"] == PLUGIN_ENTRY_POINT


def test_attested_on_is_refused_not_ignored(contract):
    """A reader that still wrote the old block must learn, at load, that it
    no longer means anything."""
    versions = {**contract["versions"],
                "attested_on": {"vllm": "0.28.0", "torch": "2.13.0", "image": "x", "device": "y"}}
    with pytest.raises(ValueError, match=r"versions carries unknown field\(s\) \['attested_on'\]"):
        validate_serving_contract(_with_versions(contract, versions))


@pytest.mark.parametrize("missing", sorted(VERSIONS_KEYS))
def test_every_versions_field_is_required(contract, missing):
    versions = {k: v for k, v in contract["versions"].items() if k != missing}
    with pytest.raises(ValueError, match=rf"versions is missing \['{missing}'\]"):
        validate_serving_contract(_with_versions(contract, versions))


def test_a_default_image_no_cell_attests_is_refused(contract):
    unattested = "example/runtime@sha256:" + "c" * 64
    versions = {**contract["versions"], "default_serve_image": unattested}
    with pytest.raises(ValueError, match="no lane_eligibility cell attests"):
        validate_serving_contract(_with_versions(contract, versions))


def test_a_floating_tag_is_refused_as_the_default_image(contract):
    versions = {**contract["versions"], "default_serve_image": "vllm/vllm-openai:latest"}
    with pytest.raises(ValueError, match="digest reference"):
        validate_serving_contract(_with_versions(contract, versions))


def test_a_wrong_entry_point_or_empty_version_is_refused(contract):
    versions = {**contract["versions"], "plugin_entry_point": "tessera = tessera:register"}
    with pytest.raises(ValueError, match="plugin_entry_point"):
        validate_serving_contract(_with_versions(contract, versions))
    versions = {**contract["versions"], "tessera": ""}
    with pytest.raises(ValueError, match="versions.tessera must be a non-empty string"):
        validate_serving_contract(_with_versions(contract, versions))
