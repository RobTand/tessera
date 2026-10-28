"""Projection preparation is real-byte CPU evidence, never a loaded vLLM route."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import tessera_construction_census as census
from tools import tessera_projection_smoke as smoke


PROJECTIONS = tuple(case for case in smoke.CASES if ".self_attn." in case[0])
RANKS = ((1, 0), (2, 0), (2, 1))


def _declaration(case, world, rank):
    _prefix, kind, columns, roles = case
    sizes = [rows for _name, rows in roles]
    declaration = {"tp_rank": rank, "tp_size": world, "input_size": columns,
                   "input_size_per_partition": columns, "output_size": sum(sizes),
                   "output_partition_sizes": sizes}
    if kind == "kda":
        declaration["replicated_shard_ids"] = [4, 5]
        partitions = [rows if index in (4, 5) else rows // world
                      for index, rows in enumerate(sizes)]
        declaration["output_partition_sizes"] = partitions
        declaration["output_size"] = sum(partitions) * world
    elif kind == "row":
        declaration["input_size_per_partition"] = columns // world
    elif kind not in ("mla_input", "replicated", "indexer"):
        declaration["output_partition_sizes"] = [rows // world for rows in sizes]
    return declaration


@pytest.fixture(scope="module")
def wires():
    from tessera.alphabet import BF16_GRID, E4M3_GRID
    from tessera.fused import pack_fused
    from tessera.unit_artifact import build_unit_artifact

    result = {}
    for case_index, case in enumerate(PROJECTIONS):
        prefix, _kind, columns, roles = case
        for label, (family, grid_name) in smoke.FAMILIES.items():
            grid = E4M3_GRID if grid_name == "E4M3" else BF16_GRID
            parts = []
            for role_index, (name, rows) in enumerate(roles):
                unit = smoke._unit(rows, columns, grid, 211 + case_index * 17 + role_index)
                _manifest, _payload, blob = build_unit_artifact(
                    unit, name, grid, 1024, fixture_id=None)
                parts.append((name, rows, blob))
            wire = pack_fused(parts)
            scheme = {"family": family, "grid": grid_name, "body": "WINDOW",
                      "plane": "CHANNEL", "q256": 1024, "rows": sum(rows for _, rows in roles),
                      "columns": columns, "wire_bytes": len(wire), "roles": list(roles)}
            result[prefix, label] = wire, scheme
    return result


@pytest.mark.parametrize("case", PROJECTIONS, ids=lambda case: case[0])
@pytest.mark.parametrize("label", smoke.FAMILIES)
@pytest.mark.parametrize("world,rank", RANKS)
def test_real_wire_role_placement_packet(wires, case, label, world, rank):
    from tools import tessera_projection_cpu as cpu

    prefix, kind, columns, roles = case
    wire, scheme = wires[prefix, label]
    declaration = _declaration(case, world, rank)
    packet = cpu.load_packet(wire, scheme, prefix, declaration,
                             wire_sha256=hashlib.sha256(wire).hexdigest())
    assert packet["wire_parsed"] is True
    assert packet["cpu_roles_decoded"] is True
    assert packet["weights_loaded"] is packet["forward_executed"] is False
    assert packet["evidence_scope"] == "cpu-fixture-only"
    assert packet["wire_bytes"] == len(wire)
    assert packet["scheme"]["family"] == smoke.FAMILIES[label][0]
    assert packet["scheme"]["roles"] == [list(role) for role in roles]
    assert packet["layer_fixture"] == declaration
    assert packet["placement"]["shard_rows"] == sum(declaration["output_partition_sizes"])
    assert packet["placement"]["shard_columns"] == declaration["input_size_per_partition"]
    assert [role["name"] for role in packet["roles"]] == [name for name, _ in roles]
    for index, (actual, (name, rows)) in enumerate(zip(packet["roles"], roles)):
        whole = kind in ("mla_input", "replicated", "indexer") or (kind == "kda" and index in (4, 5))
        row_cut = kind != "row" and not whole
        expected_rows = rows // world if row_cut else rows
        expected_columns = columns // world if kind == "row" else columns
        extent = columns if kind == "row" else rows
        size = expected_columns if kind == "row" else expected_rows
        offset = rank * size if not whole else 0
        assert actual["shape"] == [expected_rows, expected_columns]
        placement = packet["placement"]["roles"][index]
        assert (placement["extent"], placement["lo"], placement["hi"]) == (extent, offset, offset + size)
        assert actual["source_slice_equal"] is True
        assert len(actual["decoded_sha256"]) == 64
    assert json.loads(json.dumps(packet)) == packet


@pytest.mark.parametrize("rank", [0, 1])
def test_kda12576_geometry_is_not_a_loaded_wire(tmp_path, rank):
    from tools import tessera_projection_cpu as cpu

    root = Path(__file__).resolve().parents[1]
    config = root / "experiments/results/glm53_u1_stub_d_config.json"
    (tmp_path / "config.json").write_bytes(config.read_bytes())
    geometry = next(row for row in census.preflight(str(tmp_path), "fixture-only")["KDA_geometry"]
                    if row["tp_size"] == 2)
    prefix, kind, _columns, roles = PROJECTIONS[0]
    case = (prefix, kind, geometry["columns"],
            tuple((name, rows) for (name, _), rows in zip(roles, geometry["global_output_sizes"])))
    declaration = _declaration(case, 2, rank)
    scheme = {"family": "TESSERA_FP8", "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL",
              "q256": 1024, "rows": geometry["global_rows"], "columns": geometry["columns"],
              "wire_bytes": 1, "roles": list(case[3])}
    packet = cpu.placement_packet(scheme, prefix, declaration)
    assert packet["placement"]["shard_rows"] == geometry["local_rows"] == 12576
    assert packet["placement"]["shard_columns"] == 4096
    assert packet["wire_parsed"] is packet["cpu_roles_decoded"] is False
    assert packet["weights_loaded"] is packet["forward_executed"] is False
    assert "wire_sha256" not in packet
    for index in geometry["replicated_shard_ids"]:
        role = packet["placement"]["roles"][index]
        assert (role["lo"], role["hi"]) == (0, geometry["local_output_sizes"][index])


@pytest.mark.parametrize("defect", ["digest", "roles", "width", "replicas", "short_partition"])
def test_packet_refuses_bad_bytes_or_layer_declarations(wires, defect):
    from tools import tessera_projection_cpu as cpu

    case = PROJECTIONS[0]
    prefix = case[0]
    wire, original = wires[prefix, next(iter(smoke.FAMILIES))]
    scheme = copy.deepcopy(original)
    declaration = _declaration(case, 2, 1)
    digest = hashlib.sha256(wire).hexdigest()
    if defect == "digest":
        digest = "0" * 64
        reason = "sha256"
    elif defect == "roles":
        scheme["roles"] = list(reversed(scheme["roles"]))
        reason = "role|member"
    elif defect == "width":
        declaration["input_size"] += 1
        reason = "shape|tile|columns"
    elif defect == "replicas":
        declaration["replicated_shard_ids"] = [6]
        reason = "replicated_shard_ids"
    else:
        declaration["output_partition_sizes"][-1] //= 2
        reason = "shape|tile|g_a_proj"
    with pytest.raises(ValueError, match=reason):
        cpu.load_packet(wire, scheme, prefix, declaration, wire_sha256=digest)


@pytest.mark.parametrize("with_wire", [False, True])
def test_command_packet_binds_fixture_and_scheme(wires, tmp_path, with_wire):
    from tessera.serving.scheme import validate_tessera_scheme
    from tools import tessera_projection_cpu as cpu

    case = PROJECTIONS[0]
    prefix = case[0]
    wire, scheme = wires[prefix, next(iter(smoke.FAMILIES))]
    declaration = _declaration(case, 2, 1)
    scheme_path, layer_path, out = (tmp_path / name for name in ("scheme.json", "layer.json", "out.json"))
    scheme_path.write_text(json.dumps(scheme))
    layer_path.write_text(json.dumps(declaration))
    arguments = ["--scheme", str(scheme_path), "--layer-fixture", str(layer_path),
                 "--prefix", prefix, "--out", str(out)]
    if with_wire:
        wire_path = tmp_path / "weight.tessera"
        wire_path.write_bytes(wire)
        arguments.extend(["--wire", str(wire_path), "--wire-sha256", hashlib.sha256(wire).hexdigest()])
    assert cpu.main(arguments) == 0
    packet = json.loads(out.read_text())
    expected = json.loads(json.dumps(validate_tessera_scheme(scheme, prefix)))
    assert packet["scheme"] == expected
    assert packet["layer_fixture"] == declaration
    assert packet["wire_parsed"] is packet["cpu_roles_decoded"] is with_wire
    assert packet["weights_loaded"] is packet["forward_executed"] is False
