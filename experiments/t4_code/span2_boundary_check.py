"""Native span-two bitfield-boundary oracle and scoped sanitizer smoke (#995).

Fixtures are deterministic numerical test weights, not sampled model quality.
The CPU oracle reads exactly the meaningful bits and is independent of the
native byte-load implementation. GPU checks compare all nine decode states,
packed codes and scale bytes against real serialized-wire/stock decoding.
An exact SELECT prefix omits the packer's unused trailing guard for the test;
POINT keeps its actual exact prepared extent. No production planes are padded
or copied by the reader fix. Sanitizer jobs disable the caching allocator so
allocator slab slack cannot conceal an unused final-byte read.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "t8r_speed"))
from tessera.alphabet import grid_for_name
from tessera.export import TCQ_RECIPE, encode_linear
from tessera.unit_artifact import parse_unit_artifact
from tessera.lane_planes import prepare_span2_planes
from tessera.stock import materialize_stock
from tessera.kernel_a4 import (A4Unit, a4_decode_span2_tile, a4_decode_states_at,
                              native_fp4_backend)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False))


def prepare(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grid = grid_for_name("E2M1x2")
    weights = ((torch.arange(256 * 128).reshape(256, 128) % 511 - 255).float() / 1024).to(torch.bfloat16)
    entries = []
    for rate in range(1, grid.rate_cap + 1):
        recipe = TCQ_RECIPE
        unit = encode_linear(weights, grid=grid, q256=rate * 128,
                             body=recipe.body, span=recipe.span,
                             scale_plane=recipe.scale_plane, window_bits=recipe.window_bits)
        path = out / f"rate{rate}.wire"
        path.write_bytes(unit.blob)
        entries.append({"path": str(path), "offset": 0, "bytes": len(unit.blob),
                        "sha256": hashlib.sha256(unit.blob).hexdigest()})
    total = sum(e["bytes"] for e in entries)
    save(out / "data-manifest.json", {"schema": "prismaquant.prismabuild.data_manifest.v1",
         "mount_prefix": "/mnt/shared", "produced_by": {"tool": "span2_boundary_check", "unix": time.time()},
         "annotations": {"row_id": "native-span2-boundary", "phases": [{"name": "serialized-test-wires", "bytes": total, "cumulative_bytes": total}]},
         "entries": entries, "entry_count": len(entries), "total_bytes": total})


def field_value(data, bit, width):
    """Read only meaningful bits; zero-width fields dereference nothing."""
    value = 0
    for offset in range(width):
        position = bit + offset
        value = (value << 1) | ((int(data[position // 8]) >> (7 - position % 8)) & 1)
    return value


def states_reference(unit, ks, ps):
    select, label, point = (getattr(unit, name).cpu().numpy() for name in ("select", "label", "point"))
    label_lut, code = unit.label_lut.cpu().numpy(), unit.code_nibbles.cpu().numpy()
    pairs, steps, width = unit.rows // 4, unit.rows // 2, unit.rate - 1
    result = [[] for _ in range(9)]
    for k, p in zip(ks, ps):
        q = k * (pairs + 8) + 8 - unit.memory + p
        window = field_value(select, q, unit.memory + 1)
        ell = int(label_lut[window])
        stored = field_value(label, k * pairs * 2 + p * 2, 2)
        lab0, lab1 = (ell - stored) & 3, stored
        t = k * steps * width + p * 2 * width
        pt0, pt1 = field_value(point, t, width), field_value(point, t + width, width)
        values = (window, ell, stored, lab0, lab1, pt0, pt1,
                  int(code[lab0 * (1 << width) + pt0]), int(code[lab1 * (1 << width) + pt1]))
        for target, value in zip(result, values):
            target.append(value)
    return result


def check_wire(blob, rate, device):
    parsed = parse_unit_artifact(blob, device="cpu")
    prepared = prepare_span2_planes(parsed, device="cpu")
    unit = A4Unit.from_prepared(prepared)
    if (unit.rate, unit.rows, unit.cols, unit.arity, unit.half) != (rate, 256, 128, 2, 16):
        raise ValueError("fixture shape/recipe differs from the boundary oracle")
    stock = materialize_stock(parsed.unit, parsed.forests, parsed.code)
    # The meaningful SELECT plane is byte-exact; the production packer's
    # extra trailing guard is not required by any valid select bitfield.
    select_extent = unit.cols * (unit.rows // 4 + 8) // 8
    unit = replace(unit, select=unit.select[:select_extent].clone())
    ks = np.repeat(np.array([0, 1, unit.cols - 2, unit.cols - 1]), unit.rows // 4)
    ps = np.tile(np.arange(unit.rows // 4), 4)
    expected = states_reference(unit, ks, ps)
    result = {"rate": rate, "q256": rate * 128, "indexed_states": len(ks),
              "point_bits": rate - 1, "select_bits": unit.memory + 1,
              "select_bytes": unit.select.numel(), "point_bytes": unit.point.numel(),
              "wire_sha256": hashlib.sha256(blob).hexdigest()}
    if device == "cpu":
        result["status"] = "CPU inputs and exact-width numerical oracle passed; no GPU result"
        return result
    gpu = unit.to("cuda")
    actual = a4_decode_states_at(gpu, torch.as_tensor(ks, dtype=torch.int32, device="cuda"),
                               torch.as_tensor(ps, dtype=torch.int32, device="cuda"))
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got.cpu().tolist() != want:
            raise AssertionError(f"rate{rate}: numerical decode state {index} differs")
    va, vb, scales = a4_decode_span2_tile(gpu)
    va, vb = va.cpu().to(torch.int32), vb.cpu().to(torch.int32)
    packed = torch.empty((unit.rows, unit.cols // 2), dtype=torch.uint8)
    for row in range(4):
        packed[row::4] = (((va >> (4 * row)) & 15) | (((vb >> (4 * row)) & 15) << 4)).t().to(torch.uint8)
    if not torch.equal(packed, stock["weight_packed"].cpu()):
        raise AssertionError(f"rate{rate}: packed codes differ from stock wire decode")
    if not torch.equal(scales.view(torch.uint8).cpu(), stock["weight_scale"].view(torch.uint8).cpu()):
        raise AssertionError(f"rate{rate}: scale bytes differ from stock wire decode")
    result["status"] = "all native states, packed codes and scale bytes matched independent oracle"
    result["backend"] = native_fp4_backend()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", default="")
    ap.add_argument("--rates", default="1,2,3,4,5,6,7")
    ap.add_argument("--prepare-inputs", action="store_true")
    ap.add_argument("--cpu-preflight", action="store_true")
    ap.add_argument("--data-manifest", default="")
    ap.add_argument("--sanitizer", action="store_true")
    ap.add_argument("--under-sanitizer", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.prepare_inputs:
        prepare(args.out)
        return 0
    rates = [int(r) for r in args.rates.split(",")]
    if not rates or any(r not in range(1, 8) for r in rates):
        raise ValueError("rates must be actual E2M1x2 TCQ rates1..7")
    if args.sanitizer and not args.cpu_preflight and not args.under_sanitizer:
        binary = shutil.which("compute-sanitizer")
        if not binary:
            raise RuntimeError("compute-sanitizer is not installed in the declared image")
        argv = [binary, "--tool", "memcheck", "--error-exitcode", "86", "--target-processes", "all",
                sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--under-sanitizer"]
        os.execv(binary, argv)
    inputs = None
    if args.data_manifest:
        from prismabuild.client import read_data_manifest
        manifest, _encoding = read_data_manifest(args.data_manifest)
        if not args.cpu_preflight:
            from pb_staged_store import StagedInputs
            inputs = StagedInputs(args.data_manifest)
    result = {"source_commit": os.environ.get("TESSERA_HEAD"), "action_key": os.environ.get("PRISMABUILD_ACTION_KEY"),
              "source_sha256": hashlib.sha256((Path(__file__).parents[2] / "src/tessera/kernel_a4.py").read_bytes()).hexdigest(),
              "cpu_preflight": args.cpu_preflight, "sanitizer": args.under_sanitizer,
              "allocator_caching_disabled": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1", "checks": []}
    try:
        for rate in rates:
            path = str(Path(args.fixtures) / f"rate{rate}.wire")
            blob = bytes(inputs.read(path)) if inputs else Path(path).read_bytes()
            result["checks"].append(check_wire(blob, rate, "cpu" if args.cpu_preflight else "cuda"))
            save(Path(args.out) / "boundary-oracle.json", result)
        result["status"] = "passed"
        save(Path(args.out) / "boundary-oracle.json", result)
        print(json.dumps(result), flush=True)
    finally:
        if inputs:
            inputs.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
