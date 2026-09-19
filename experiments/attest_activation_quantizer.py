"""Generate ``runtime_contract.json``'s ``activation_quantizers`` block (#484).

The contract publishes an activation contract as a NAME, and the route hands
the static global scale to vLLM's compiled ``scaled_fp4_quant``.  So what a
value becomes is the RUNTIME's answer, and a producer that re-implements it is
asserting a runtime behaviour.  This script asks the kernel instead: it feeds
the probe groups ``tessera.serving.activation_attestation`` constructs, reads
the codes and block-scale bytes back, and writes them into the contract.

The inputs are the repository's and the outputs are the kernel's.  Nothing
here chooses a rounding rule, a tie-break or a saturation behaviour; every
such value in the emitted table came out of the operator on this device.

Run it in the image the platform's cells attest, with a GPU:

    python3 experiments/attest_activation_quantizer.py emit \\
        --platform sm_121 --image <the platform's serve_image digest> \\
        --out activation_quantizers.json
    python3 experiments/attest_activation_quantizer.py verify \\
        --platform sm_121                      # against the packaged contract

``verify`` is the check that the packaged table is still this runtime's
answer; it is what re-runs when the pinned vLLM moves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.serving.activation_attestation import (  # noqa: E402
    ACTIVATION_QUANTIZER_SCHEMA,
    PROBES,
    bf16_value,
    validate_attestation,
)

#: This script's own repository path, which the block publishes so a reader
#: knows what to re-run.  Relative, because the packaged document may not name
#: a path only one fleet can open (``tests/test_contract_is_portable.py``).
GENERATOR = "experiments/attest_activation_quantizer.py"

#: Which operator answers for each activation contract.  The route module is
#: the owner of that binding; this is the contract-name half of it.
OPERATORS = {
    "e2m1_group16_ue4m3_static": {
        "op": "torch.ops._C.scaled_fp4_quant",
        "unit": "group", "unit_length": 16, "grid": "E2M1",
        "block_scale": "UE4M3", "global_scale": "static_per_module",
    },
}

#: The cuBLAS scale plane is tiled 128 rows deep, so a probe matrix is padded
#: to a whole tile.  The padding rows are copies of the first probe and the
#: generator checks they decode identically, which is a free proof that the
#: de-swizzle below put every group back where it came from.
ROW_TILE = 128


def _decode(packed, blocked, rows, columns):
    """Raw E2M1 nibbles and raw UE4M3 block-scale bytes out of the native layout.

    The permutation is asked of the route's own ``blocked_scales`` rather than
    restated here -- the same move ``experiments/bench_native_operator.py``
    ``represented_native_input`` makes, for the same reason: the swizzle is the
    hardware's layout and this file may not own a second copy of it.  An
    unswizzled plane is silently miscomputed by ``_scaled_mm`` (67-70 %), so a
    decode that guessed would be the 2026-08-17 failure again.
    """
    import torch
    from tessera.serving.nvfp4_route import GROUP_SIZE, blocked_scales

    groups = columns // GROUP_SIZE
    ids = torch.arange(1, rows * groups + 1, device=packed.device).reshape(rows, groups)
    permutation = blocked_scales(ids)
    valid = permutation > 0
    if int(valid.sum()) != rows * groups:
        raise ValueError("the scale-plane round trip does not cover every group")
    flat = blocked.reshape(-1).view(torch.uint8)
    stored = torch.empty(ids.numel(), dtype=torch.uint8, device=packed.device)
    stored[permutation[valid] - 1] = flat[valid]
    # The inverse must actually be the inverse: re-swizzling the recovered
    # plane has to reproduce the bytes the kernel wrote, at every valid slot.
    again = blocked_scales(stored.reshape(rows, groups).view(torch.float8_e4m3fn))
    if not bool((again.view(torch.uint8)[valid] == flat[valid]).all()):
        raise ValueError("the recovered scale plane does not re-swizzle to the kernel's")
    bytes_ = packed.view(torch.uint8).reshape(rows, columns // 2)
    codes = torch.stack((bytes_ & 15, bytes_ >> 4), dim=-1).reshape(rows, columns)
    return codes.to(torch.int64).cpu().tolist(), stored.reshape(rows, groups).cpu().tolist()


def _run_probes(probes):
    """One kernel call per distinct global scale; one probe per matrix row."""
    import torch
    from tessera.serving import native_ops

    # ``has_fp4_quant`` rather than ``require_native_fp4_quant``: the latter
    # also asks the packaged contract whether this PLATFORM is backed, and the
    # packaged contract is what this script writes.  The generator needs the
    # operator to exist, not the lane to be admitted -- and a serving
    # admission decided by a table that does not exist yet is a bootstrap this
    # script would never escape.  Import the namespace the same way the route
    # does, then probe it by name.
    native_ops._load_native_ops("activation quantizer attestation")
    if not native_ops.has_fp4_quant():
        raise SystemExit(
            "this build registers no torch.ops._C.scaled_fp4_quant; the fp4 "
            "activation quantizer cannot be attested from a runtime that does "
            "not carry it")
    results = {}
    scales = sorted({p.global_scale for p in probes})
    for scale in scales:
        rows = [p for p in probes if p.global_scale == scale]
        values = [[bf16_value(b) for b in p.input_bits] for p in rows]
        padded = values + [values[0]] * (-len(values) % ROW_TILE)
        x = torch.tensor(padded, dtype=torch.bfloat16, device="cuda").contiguous()
        g = torch.tensor([scale], dtype=torch.float32, device="cuda")
        packed, blocked = native_ops.native_fp4_quant(x, g)
        codes, stored = _decode(packed, blocked, x.shape[0], x.shape[1])
        for index in range(len(values), len(padded)):
            if codes[index] != codes[0] or stored[index] != stored[0]:
                raise ValueError("a repeated padding row decoded differently from its original")
        for index, probe in enumerate(rows):
            if len(stored[index]) != 1:
                raise ValueError("one probe row must be exactly one NVFP4 group")
            results[probe.identifier] = {
                **probe.as_json(),
                "stored_scale": int(stored[index][0]),
                "codes": [int(c) for c in codes[index]],
            }
    return [results[p.identifier] for p in probes]


def _generated(image):
    import torch

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True).splitlines()[0].strip()
    import importlib.metadata
    with (Path(__file__).resolve().parents[1] / GENERATOR).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "image": image,
        "vllm": importlib.metadata.version("vllm"),
        "torch": torch.__version__,
        "device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "driver": driver,
        "generator_sha256": digest,
    }


def _block(platform, image, contract_name):
    return {
        "schema": ACTIVATION_QUANTIZER_SCHEMA,
        "generator": GENERATOR,
        "platforms": {
            platform: {
                "generated": _generated(image),
                "contracts": {contract_name: {**OPERATORS[contract_name],
                                              "vectors": _run_probes(PROBES)}},
            },
        },
    }


def _packaged():
    """The packaged document's BYTES, not a validated contract.

    ``load_serving_contract`` validates, and validation now requires the very
    block this script produces, so going through it would make the first table
    unbuildable.  The generator reads the lane table it needs and hands its own
    output to ``validate_activation_quantizers`` directly.
    """
    from tessera.serving.contract import contract_path
    return json.loads(contract_path().read_text())


def _lane_facts(contract):
    lane = contract["lane_eligibility"]
    served = {}
    for cell in lane["cells"]:
        served.setdefault(cell["platform"], set()).add(cell["activation_contract"])
    return sorted(lane["platforms"]), served


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("emit", "verify"):
        one = sub.add_parser(name)
        one.add_argument("--platform", required=True)
        one.add_argument("--contract", default="e2m1_group16_ue4m3_static")
        one.add_argument("--image",
                         help="the attestation this run checks (verify) or writes "
                              "(emit). Emit requires it: a table without the image "
                              "it was taken under is an assertion again. Verify "
                              "defaults to every attestation the platform "
                              "publishes and passes when this runtime reproduces "
                              "any one of them.")
        if name == "emit":
            one.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.contract not in OPERATORS:
        parser.error(f"no operator is bound to activation contract {args.contract!r}")
    if args.command == "emit" and not args.image:
        parser.error("emit requires --image, the digest reference of the image this runs in")

    packaged = _packaged()
    platforms, served = _lane_facts(packaged)
    image = args.image
    if image is None:
        platform_entry = packaged["activation_quantizers"]["platforms"][args.platform]
        if not isinstance(platform_entry, list) or not platform_entry:
            print(f"the packaged contract's {args.platform} entry is not a list of one "
                  "attestation per image; this verifier reads the v2 grammar only",
                  file=sys.stderr)
            return 2
        image = platform_entry[0]["generated"]["image"]
    block = _block(args.platform, image, args.contract)

    from tessera.serving.contract import require_runtime_image
    validate_attestation(block["platforms"][args.platform],
                         served.get(args.platform, set()),
                         require_image=require_runtime_image,
                         at=(f"emitted {args.platform} attestation"
                             if args.command == "emit"
                             else f"fresh {args.platform} table"))

    if args.command == "emit":
        Path(args.out).write_text(json.dumps(block, indent=1) + "\n")
        print(f"wrote {args.out}")
        return 0
    published = packaged.get("activation_quantizers")
    if published is None:
        print("the packaged contract publishes no activation_quantizers block", file=sys.stderr)
        return 2
    entries = published["platforms"].get(args.platform, [])
    if not isinstance(entries, list):
        print(f"the packaged contract's {args.platform} entry is not a list of one "
              "attestation per image; this verifier reads the v2 grammar only",
              file=sys.stderr)
        return 2
    if args.image is not None:
        entries = [entry for entry in entries
                   if entry["generated"]["image"] == args.image]
        if not entries:
            print(f"the packaged contract attests no {args.platform} table generated on "
                  f"{args.image}", file=sys.stderr)
            return 2
    matched, differing = [], {}
    for entry in entries:
        old = entry.get("contracts", {}).get(args.contract)
        if old is None:
            continue
        fresh = block["platforms"][args.platform]["contracts"][args.contract]
        if old["vectors"] == fresh["vectors"]:
            matched.append(entry["generated"]["image"])
        else:
            differing[entry["generated"]["image"]] = [
                f["id"] for f, o in zip(fresh["vectors"], old["vectors"]) if f != o]
    if matched:
        print(f"{len(block['platforms'][args.platform]['contracts'][args.contract]['vectors'])} "
              f"vectors reproduce the packaged table generated on {matched[0]} "
              f"({block['platforms'][args.platform]['generated']['device']})")
        return 0
    if not differing:
        print(f"the packaged contract attests nothing for {args.platform}/{args.contract}",
              file=sys.stderr)
        return 2
    first, ids = next(iter(differing.items()))
    print(f"this runtime no longer emits the packaged table generated on {first}; "
          f"regenerate it. differing vectors: {ids or 'the vector set itself moved'}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
