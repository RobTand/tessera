"""The minor <= 6 checkpoint the contract's evidence was measured on, after
schema minor 7 (tessera#144).

Two claims, kept apart because they are different claims:

1. **Old bytes still load** through the serving plugin's own parse: every
   module of the checkpoint, through ``parse_tessera_blob_for_scheme`` with
   the checkpoint's own sidecar scheme, to units the reader calls LEGACY.
   (``tests/test_kernel_window*.py`` decode the same units through the window
   kernel against the stock twin; ``test_slice_unit`` rebuilds them byte for
   byte with the layout they were written in.)

2. **New bytes are the same payload** -- in two halves, because they are
   owned by different things.  The half this change owns: one unit
   re-encoded from the source model under this tree's encoder, the way
   ``experiments/export_tessera_serving.py``'s dense path encodes it, laid
   out LEGACY and LADDER, is one plane region and one ``payload_digest``;
   what differs is the envelope -- the header minor and the COMPLETION
   descriptor's counts -- and nothing the serve decodes.  The half the
   encoder owns -- that the re-encode equals the bytes ON DISK -- does not
   hold today, on this tree or on the tree before it; it is recorded as a
   strict xfail carrying the numbers (see the last test), so it is a fact
   about the encoder since ``8070ec6``, not about the layout.

The checkpoint is ``gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook`` (E4M3,
q256 1024, window body on the CHANNEL plane): the artifact behind
``docs/measurements/tessera-serving-plugin-2026-09-02.md`` and the other
receipts ``runtime_contract.json``'s ``lane_eligibility`` cells cite.
"""
from __future__ import annotations

import json

import pytest
import torch

import box_artifacts
from tessera.container import parse
from tessera.fused import pack_fused, parse_fused
from tessera.planes import PlaneKind, PlaneLayout
from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact, read_unit_artifact

CHECKPOINT = ("gbfam", "qwen3-0.6b-tessera-e4m3-reach-gridbook")
REACH = box_artifacts.path("runs", *CHECKPOINT)
QWEN = box_artifacts.path("models", "Qwen3-0.6B", "model.safetensors")
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the encoder is a CUDA path"
)


def _config_groups() -> dict:
    config = json.loads((REACH / "config.json").read_text())
    return config["quantization_config"]["config_groups"]


def _sidecar_totals() -> dict:
    return json.loads((REACH / "tessera_gridbook_manifest.json").read_text())["totals"]


def _group_for(target: str) -> dict:
    (group,) = [g for g in _config_groups().values() if g["targets"] == [target]]
    return group


def _wire_bytes(target: str) -> bytes:
    from safetensors import safe_open

    with safe_open(str(REACH / "model.safetensors"), framework="pt") as handle:
        return bytes(handle.get_tensor(target + ".wire_bytes").numpy().tobytes())


@box_artifacts.require("runs", *CHECKPOINT, "model.safetensors")
def test_every_module_of_the_shipped_checkpoint_loads_through_the_plugins_parse():
    """Claim 1.  The plugin's load-time parse, module by module, against the
    sidecar the checkpoint itself carries; every unit reads as the layout it
    was written in, at a minor this reader still lists."""
    from safetensors import safe_open

    from tessera.container import SCHEMA_MINORS_READ
    from tessera.serving.scheme import parse_tessera_blob_for_scheme

    groups = _config_groups()
    totals = _sidecar_totals()
    assert len(groups) == totals["modules"]
    units, minors = 0, set()
    with safe_open(str(REACH / "model.safetensors"), framework="pt") as handle:
        for group in groups.values():
            (target,) = group["targets"]
            blob = bytes(handle.get_tensor(target + ".wire_bytes").numpy().tobytes())
            parsed = parse_tessera_blob_for_scheme(blob, group["scheme"], target, device="cpu")
            assert [role for role, _unit in parsed] == [
                role for role, _rows in group["scheme"]["roles"]
            ]
            for member, (_role, unit) in zip(parse_fused(blob), parsed):
                assert unit.manifest.layout is PlaneLayout.LEGACY
                assert member.blob[10] < 7 and member.blob[10] in SCHEMA_MINORS_READ
                minors.add(member.blob[10])
                wire = unit.manifest.plane_order
                assert unit.manifest.terminals[0].plane_elements[
                    wire.index(PlaneKind.COMPLETION)
                ] == 0
                units += 1
    assert units == totals["units"]
    assert minors, "no unit parsed"


def _shipped_unit_and_fresh_encode():
    """The on-disk ``model.layers.0.mlp.down_proj`` unit, and the same weight
    encoded by this tree exactly as the exporter's dense path encodes it (no
    Hessian, the recipe's defaults, verify on)."""
    from safetensors import safe_open

    from tessera.alphabet import SERIALISABLE_GRIDS
    from tessera.export import encode_linear_planes

    target = "model.layers.0.mlp.down_proj"
    scheme = _group_for(target)["scheme"]
    assert (scheme["grid"], scheme["q256"], scheme["body"], scheme["plane"]) == (
        "E4M3", 1024, "WINDOW", "CHANNEL"
    )
    disk = _wire_bytes(target)
    assert len(disk) == scheme["wire_bytes"]
    (member,) = parse_fused(disk)
    with safe_open(str(QWEN), framework="pt") as handle:
        weight = handle.get_tensor(target + ".weight").to("cuda", torch.float32).contiguous()
    grid = next(g for g in SERIALISABLE_GRIDS.values() if g.name == scheme["grid"])
    exported, unit, forests = encode_linear_planes(
        weight, grid=grid, q256=scheme["q256"], name=target
    )
    return target, scheme, member, grid, exported, unit, forests


@needs_cuda
@box_artifacts.require("runs", *CHECKPOINT, "model.safetensors")
@box_artifacts.require("models", "Qwen3-0.6B", "model.safetensors")
def test_the_two_layouts_write_one_payload_for_a_shipped_units_weight():
    """Claim 2, the half this change owns.  The same encode of a shipped
    unit's weight, laid out LEGACY and LADDER, is one plane region and one
    ``payload_digest``; the envelopes differ by the header minor and the
    COMPLETION descriptor's counts and nothing else; the minor-7 unit
    censuses through the plugin's parse and decodes to the LEGACY one."""
    from tessera.export import DEFAULT_CODE
    from tessera.serving.scheme import parse_tessera_blob_for_scheme

    target, scheme, member, grid, exported, unit, forests = _shipped_unit_and_fresh_encode()
    fresh = parse(exported.blob)
    assert fresh.manifest.layout is PlaneLayout.LADDER and exported.blob[10] == 7
    _m, legacy_region, legacy_blob = build_unit_artifact(
        unit, target, forests, scheme["q256"] * grid.arity, DEFAULT_CODE,
        layout=PlaneLayout.LEGACY,
    )
    legacy = parse(legacy_blob)
    assert legacy.manifest.layout is PlaneLayout.LEGACY and legacy_blob[10] < 7
    assert legacy_region == fresh.plane_region == legacy.plane_region
    assert legacy.manifest.payload_digest == fresh.manifest.payload_digest
    assert legacy.terminal.payload_digest == fresh.terminal.payload_digest
    assert legacy.terminal.exact_bytes == fresh.terminal.exact_bytes
    assert dict(zip(fresh.manifest.plane_order, fresh.terminal.plane_elements)) == dict(
        zip(legacy.manifest.plane_order, legacy.terminal.plane_elements)
    )
    assert fresh.terminal.plane_elements[
        fresh.manifest.plane_order.index(PlaneKind.COMPLETION)
    ] == 0
    assert len(exported.blob) < len(legacy_blob)
    assert fresh.manifest.encoder_fixture_id == legacy.manifest.encoder_fixture_id
    # Census through the plugin's parse, under the scheme a fresh export's
    # sidecar would write (``wire_bytes`` is the container's own length).
    container = pack_fused([(member.name, member.rows, exported.blob)])
    [(role, parsed)] = parse_tessera_blob_for_scheme(
        container, dict(scheme, wire_bytes=len(container)), target, device="cuda"
    )
    assert role == member.name and parsed.manifest.layout is PlaneLayout.LADDER
    assert torch.equal(
        read_unit_artifact(exported.blob, device="cuda"),
        read_unit_artifact(legacy_blob, device="cuda"),
    )


@needs_cuda
@box_artifacts.require("runs", *CHECKPOINT, "model.safetensors")
@box_artifacts.require("models", "Qwen3-0.6B", "model.safetensors")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "not the layout (measured 2026-09-05 on sparklina): origin/master "
        "5bec67b, the tree before schema minor 7, calling the exporter's dense "
        "path with its defaults and no Hessian, re-encodes this unit to a "
        "region differing from the disk in 63952 of 1591296 bytes, first at "
        "16838 (BODY), last at 1591290 (DIAG_SV), payload_digest unequal -- "
        "and this tree differs at the same bytes.  The likely cause is the "
        "encoder moving since 8070ec6 (the unit is minor 3 with no reach "
        "record; the reach-aware start became the default the same day); the "
        "invocation that wrote the checkpoint is not recorded, so that is not "
        "proven.  Turns green when the checkpoint is re-exported by the "
        "encoder that ships, or an encode reproduces the disk; strict, so "
        "either event is noticed."
    ),
)
def test_a_shipped_unit_re_encoded_by_this_tree_is_the_on_disk_payload():
    """Claim 2, the half the encoder owns: this tree's encode of the shipped
    unit's weight against the bytes on disk (written 2026-09-02 at
    ``8070ec6``, minor 3, no reach record).  The assertion names the first
    differing byte and the plane it falls in, so a failure here is a fact
    about the encoder's drift since that commit, attributable by running the
    same comparison on the tree before schema minor 7 -- not a fact about the
    layout, which the test above isolates.  It is, today: see the xfail."""
    from tessera.container import plane_ranges

    _target, _scheme, member, _grid, exported, _unit, _forests = _shipped_unit_and_fresh_encode()
    fresh, old = parse(exported.blob), parse(member.blob)
    assert old.manifest.layout is PlaneLayout.LEGACY and member.blob[10] < 7
    assert dict(zip(fresh.manifest.plane_order, fresh.terminal.plane_elements)) == dict(
        zip(old.manifest.plane_order, old.terminal.plane_elements)
    )
    a, b = fresh.plane_region, old.plane_region
    diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    where = {
        d.kind.name: (off, off + total)
        for d, off, _content, total in plane_ranges(old.manifest, old.terminal) if total
    }
    inside = [k for k, (lo, hi) in where.items() if diffs and lo <= diffs[0] < hi]
    assert a == b, (
        f"{len(diffs)} of {len(b)} region bytes differ from the on-disk unit, "
        f"first at {diffs[:1]} (in {inside}), last at {diffs[-1:]}; "
        f"payload_digest equal: {fresh.manifest.payload_digest == old.manifest.payload_digest}"
    )
    assert fresh.manifest.payload_digest == old.manifest.payload_digest
    assert torch.equal(
        read_unit_artifact(exported.blob, device="cuda"),
        read_unit_artifact(member.blob, device="cuda"),
    )
