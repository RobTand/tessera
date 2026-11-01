"""Verify every tensor in the derived BF16 stock checkpoint.

The reference comes from the wire reader and the stock renderer. It has one
BF16 conversion per weight. The Tessera route instead keeps raw BF16 values
and FP32 row scales separate through the dot. These are distinct arithmetic
contracts over the same encode.

The check also derives the stock tensor from the streamed pair. It reads only
checkpoint bytes. With ``--source``, it verifies tensor names, shapes, BF16
dtypes and the absence of a quantization configuration.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.bf16_route import prepare_bf16_unit, stream_bf16  # noqa: E402
from tessera.stock import materialize_stock  # noqa: E402
from tessera.fused import parse_fused  # noqa: E402
from tessera.unit_artifact import parse_unit_artifact  # noqa: E402


def open_all(directory: Path):
    handles, index = [], {}
    for path in sorted(directory.glob("*.safetensors")):
        handle = safe_open(str(path), framework="pt")
        handles.append(handle)
        for key in handle.keys():
            index[key] = handle
    return handles, index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wire", type=Path, required=True)
    ap.add_argument("--twin", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=None,
                    help="the BF16 checkpoint the export read; enables the "
                         "structural check (names, shapes, dtypes)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--streamed-every", type=int, default=8,
                    help="also run the streamed decoder on every Nth unit "
                         "(it is the same tensor; this bounds the cost)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # The exporter writes ``tessera_serving_manifest.json``; this reader was
    # the last one left asking for the retired Gridbook lane's spelling
    # (2026-09-02), so every check against a current export died on a missing
    # file rather than on anything it was written to catch. The old name is
    # still accepted because the 2026-09-02 BF16 artifacts carry it and they
    # are still the ones a historical comparison reads.
    for name in ("tessera_serving_manifest.json", "tessera_gridbook_manifest.json"):
        candidate = args.wire / name
        if candidate.is_file():
            manifest = json.loads(candidate.read_text())
            break
    else:
        raise SystemExit(
            f"{args.wire}: no tessera_serving_manifest.json (nor the retired "
            "tessera_gridbook_manifest.json); the twin check reads the roles the "
            "export WROTE, so there is nothing here to check against")
    _wh, wire_index = open_all(args.wire)
    _th, twin_index = open_all(args.twin)
    twin_config = json.loads((args.twin / "config.json").read_text())
    problems: list[str] = []

    structure: dict[str, object] = {"checked": False}
    if args.source is not None:
        _sh, src_index = open_all(args.source)
        src_keys, twin_keys = set(src_index), set(twin_index)
        shape_bad, dtype_bad = [], []
        for key in sorted(src_keys & twin_keys):
            a, b = src_index[key].get_slice(key), twin_index[key].get_slice(key)
            if list(a.get_shape()) != list(b.get_shape()):
                shape_bad.append(key)
            if b.get_dtype() != "BF16":
                dtype_bad.append(f"{key}:{b.get_dtype()}")
        structure = {
            "checked": True,
            "source_tensors": len(src_keys),
            "twin_tensors": len(twin_keys),
            "missing_from_twin": sorted(src_keys - twin_keys)[:8],
            "extra_in_twin": sorted(twin_keys - src_keys)[:8],
            "shape_mismatches": shape_bad[:8],
            "non_bf16_tensors": dtype_bad[:8],
        }
        if src_keys != twin_keys or shape_bad or dtype_bad:
            problems.append(f"twin is not structurally the source: {structure}")
        for handle in _sh:
            del handle

    checked = mismatched = streamed_checked = streamed_bad = 0
    worst = None
    started = time.time()
    for module, record in manifest["modules"].items():
        if record["family"] != "TESSERA_BF16":
            continue
        key = f"{module}.wire_bytes"
        blob = bytes(wire_index[key].get_tensor(key).numpy().tobytes())
        members = parse_fused(blob)
        names = [r["tensor"] for r in record["roles"]]
        if len(members) != len(names):
            problems.append(
                f"{module}: {len(members)} framed roles for {len(names)} recorded"
            )
            continue
        for member, name in zip(members, names):
            parsed = parse_unit_artifact(member.blob, device=args.device)
            tile = materialize_stock(parsed.unit, parsed.grid, parsed.code)["weight"]
            got = twin_index[name].get_tensor(name).to(args.device)
            checked += 1
            if got.dtype is not torch.bfloat16 or not torch.equal(got, tile):
                mismatched += 1
                delta = float((got.float() - tile.float()).abs().max())
                problems.append(f"{name}: twin differs from the derived stock tile, max |d| {delta}")
                worst = max(worst or 0.0, delta)
            if checked % args.streamed_every == 0:
                streamed_checked += 1
                values, scale = stream_bf16(prepare_bf16_unit(parsed.unit))
                derived = (values.float() * scale[:, None]).to(torch.bfloat16)
                if not torch.equal(derived, tile):
                    streamed_bad += 1
                    problems.append(f"{name}: streamed decode != tile")
            del parsed, tile, got
        if checked % 40 == 0:
            print(f"  [{checked}] {time.time() - started:.0f}s", flush=True)

    out = {
        "wire": str(args.wire), "twin": str(args.twin),
        "units_checked": checked, "units_mismatched": mismatched,
        "streamed_checked": streamed_checked, "streamed_mismatched": streamed_bad,
        "worst_abs_diff": worst,
        "twin_has_quantization_config": "quantization_config" in twin_config,
        "structure": structure,
        "wire_bpp": manifest["totals"]["wire_bpp"],
        "on_disk_bpp": manifest["totals"]["on_disk_bpp"],
        "resident_mode_bpp": manifest["totals"]["resident_mode_bpp"],
        "quantized_params": manifest["totals"]["quantized_params"],
        "checkpoint_bytes": manifest["totals"]["checkpoint_bytes"],
        "passthrough_bytes": manifest["totals"]["passthrough_bytes"],
        "problems": problems[:20],
        "secs": time.time() - started,
    }
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    if mismatched or streamed_bad or not checked or problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
