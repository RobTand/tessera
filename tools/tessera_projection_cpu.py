#!/usr/bin/env python3
"""Build projection preparation packets from explicit CPU layer fixtures.

These are wire-reader and shard-placement checks, not vLLM load, route, graph,
quality or timing qualification. Layer declarations are supplied, not inferred
from a projection name or from the wire's shape.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from tessera.serving.scheme import parse_tessera_blob_for_scheme, validate_tessera_scheme
from tessera.serving.sharding import AXIS_COLUMNS, AXIS_ROWS, plan_shard_for_layer, shard_parsed_roles


def _fixture_plan(scheme, prefix, declaration):
    declared = validate_tessera_scheme(scheme, prefix)
    plan = plan_shard_for_layer(
        prefix, SimpleNamespace(**declaration), roles=declared["roles"], columns=declared["columns"],
        input_size_per_partition=declaration["input_size_per_partition"],
        output_partition_sizes=declaration["output_partition_sizes"],
        input_size=declaration["input_size"], output_size=declaration["output_size"])
    return declared, plan


def _packet(declared, plan, declaration):
    placement = asdict(plan)
    placement["roles"] = list(placement["roles"])
    return {"schema": "tessera.projection-cpu-fixture.v1", "evidence_scope": "cpu-fixture-only",
            "weights_loaded": False, "forward_executed": False, "wire_parsed": False,
            "cpu_roles_decoded": False, "placement": placement,
            "scheme": json.loads(json.dumps(declared)),
            "layer_fixture": json.loads(json.dumps(declaration))}


def placement_packet(scheme, prefix, declaration):
    """Check full-size geometry without claiming that any weight bytes exist."""
    declared, plan = _fixture_plan(scheme, prefix, declaration)
    return _packet(declared, plan, declaration)


def load_packet(wire, scheme, prefix, declaration, *, wire_sha256):
    """Read real bytes and compare each decoded local role to its parent slice."""
    import torch
    from tessera.decode import reconstruct_unit

    digest = hashlib.sha256(wire).hexdigest()
    if digest != wire_sha256:
        raise ValueError(f"{prefix}: wire sha256 differs from the fixture declaration")
    parsed = parse_tessera_blob_for_scheme(wire, scheme, prefix, device="cpu")
    declared, plan = _fixture_plan(scheme, prefix, declaration)
    local = shard_parsed_roles(parsed, plan)
    packet = _packet(declared, plan, declaration)
    roles = []
    for (name, parent), (local_name, child) in zip(parsed, local, strict=True):
        if local_name != name:
            raise ValueError(f"{prefix}: local role {local_name!r} differs from parent role {name!r}")
        source = reconstruct_unit(parent.unit, parent.forests, parent.code)
        decoded = reconstruct_unit(child.unit, child.forests, child.code)
        role = plan.role(name)
        if plan.axis == AXIS_ROWS:
            source = source[role.lo:role.hi, :]
        elif plan.axis == AXIS_COLUMNS:
            source = source[:, role.lo:role.hi]
        if not torch.equal(source, decoded):
            raise ValueError(f"{prefix}: role {name!r} CPU decode differs from its source slice")
        raw = decoded.contiguous().view(torch.uint8).numpy().tobytes()
        roles.append({"name": name, "shape": list(decoded.shape), "dtype": str(decoded.dtype),
                      "source_slice_equal": True, "decoded_sha256": hashlib.sha256(raw).hexdigest()})
    packet.update(wire_parsed=True, cpu_roles_decoded=True, wire_bytes=len(wire),
                  wire_sha256=digest, roles=roles)
    return packet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", type=Path, required=True)
    parser.add_argument("--layer-fixture", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--wire", type=Path)
    parser.add_argument("--wire-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.wire is None) != (args.wire_sha256 is None):
        parser.error("--wire and --wire-sha256 must be supplied together")
    scheme = json.loads(args.scheme.read_text())
    declaration = json.loads(args.layer_fixture.read_text())
    if args.wire is None:
        packet = placement_packet(scheme, args.prefix, declaration)
    else:
        packet = load_packet(args.wire.read_bytes(), scheme, args.prefix, declaration,
                             wire_sha256=args.wire_sha256)
    args.out.write_text(json.dumps(packet, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
