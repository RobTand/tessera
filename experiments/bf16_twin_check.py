"""Does the stock twin hold exactly what the wire decodes to?  Every unit.

The twin is the artifact vanilla vLLM serves, and the wire is the artifact the
lane will serve.  A served comparison between them is only a comparison of two
*servings* of one encode if the two really are one encode -- so this reads the
wire back from bytes, materialises it, and asserts bit equality with the twin
tensor, per unit, with no tolerance.  It also re-checks the streamed decoder
against the same tensor, because that is the path the lane's product mode
takes and it must not be a third rendering.

It reads the checkpoints and nothing else: no encoder state is carried over
from the export, which is what makes this a check rather than a restatement.
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

from tessera.bf16_route import prepare_bf16_unit, stream_bf16_tile  # noqa: E402
from tessera.decode import materialize_bf16  # noqa: E402
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--streamed-every", type=int, default=8,
                    help="also run the streamed decoder on every Nth unit "
                         "(it is the same tensor; this bounds the cost)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    manifest = json.loads((args.wire / "tessera_gridbook_manifest.json").read_text())
    _wh, wire_index = open_all(args.wire)
    _th, twin_index = open_all(args.twin)
    twin_config = json.loads((args.twin / "config.json").read_text())

    checked = mismatched = streamed_checked = streamed_bad = 0
    worst = None
    started = time.time()
    problems: list[str] = []
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
            tile = materialize_bf16(parsed.unit, parsed.grid, parsed.code)
            got = twin_index[name].get_tensor(name).to(args.device)
            checked += 1
            if got.dtype is not torch.bfloat16 or not torch.equal(got, tile):
                mismatched += 1
                delta = float((got.float() - tile.float()).abs().max())
                problems.append(f"{name}: twin != materialize_bf16, max |d| {delta}")
                worst = max(worst or 0.0, delta)
            if checked % args.streamed_every == 0:
                streamed_checked += 1
                if not torch.equal(stream_bf16_tile(prepare_bf16_unit(parsed.unit)), tile):
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
    if mismatched or streamed_bad or not checked:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
