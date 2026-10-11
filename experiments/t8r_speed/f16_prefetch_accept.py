"""Validate selected f16 probe stages. Success never closes issue 739."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bench_pair_scope import requested_cells
from finalize_native_build import binding, verify_finalized

ARMS = ("base", "new", "newb", "baseb")
ISA_REFERENCE = "https://docs.nvidia.com/cuda/archive/13.0.3/cuda-binary-utilities/index.html#blackwell-instruction-set"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def digest(path):
    return binding(Path(path))["sha256"]


def native_identity(root, arm, doc, source):
    source = Path(source).resolve()
    meta = doc["meta"]
    native = meta["native"]
    path = Path(native["path"]).resolve()
    if not path.is_relative_to(root / arm / "home/torch_extensions"):
        raise ValueError(f"{arm}: selected ELF is outside its own cache")
    source_sha = digest(source)
    if (meta["library"] != "e4m3" or meta["kernel_sha"] != source_sha
            or Path(native["source"]).resolve() != source or native["source_sha256"] != source_sha):
        raise ValueError(f"{arm}: imported source differs from requested arm")
    if digest(path) != native["sha256"] or not native["executable_mappings"]:
        raise ValueError(f"{arm}: loaded ELF identity is incomplete or changed")
    after = meta["native_after"]
    if (any(after[key] != native[key] for key in ("path", "sha256", "source", "source_sha256", "pid"))
            or not after["executable_mappings"]):
        raise ValueError(f"{arm}: native identity changed before the final fence")
    finalization = Path(meta["finalization_path"])
    if digest(finalization) != meta["finalization_sha256"]:
        raise ValueError(f"{arm}: finalization receipt changed")
    verify_finalized(finalization, source, path, native["sha256"])
    return native


def cells(doc, expected, arm, *, ncu=False):
    meta = doc["meta"]
    declared = {(case, mode, m) for case in meta["cases"] for mode in meta["modes"] for m in meta["ms"]}
    if declared != expected:
        raise ValueError(f"{arm}: recorded request differs from requested population")
    result = {}
    for row in doc["results"]:
        for raw_m, cell in row["cells"].items():
            key = (row["case"], row["mode"], int(raw_m))
            if key in result:
                raise ValueError(f"{arm}: duplicate cell {key}")
            if ncu:
                if cell.get("ncu") is not True:
                    raise ValueError(f"{arm}: missing NCU cell {key}")
            else:
                if not re.fullmatch("[0-9a-f]{64}", cell.get("out_sha256", "")):
                    raise ValueError(f"{arm}: missing output hash {key}")
                median = cell["wall"]["median_ms"]
                if type(median) not in (int, float) or not math.isfinite(median) or median <= 0:
                    raise ValueError(f"{arm}: invalid timing {key}")
            result[key] = cell
    if set(result) != expected:
        raise ValueError(f"{arm}: cell population differs: missing={sorted(expected-set(result))}, extra={sorted(set(result)-expected)}")
    return result


def disassembly(ptx, sass):
    """Count decoded opcodes, not the English spelling of a PTX hint."""
    counts = Counter()
    candidates = []
    with Path(sass).open(errors="replace") as stream:
        for line in stream:
            match = re.match(r"\s*/\*[0-9a-fA-F]+\*/\s*(?:@!?[UP][0-9]+\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)\s.*;", line)
            if match:
                opcode = match[1]
                counts[opcode] += 1
                if opcode.split(".")[0] in ("CCTL", "CCTLL", "CCTLT", "UTMACCTL"):
                    candidates.append(line.rstrip())
    ptx_count = None
    if ptx is not None:
        with Path(ptx).open(errors="replace") as stream:
            ptx_count = sum(bool(re.match(r"\s*(?:@!?%p\d+\s+)?prefetch(?:\.[A-Za-z0-9_:]+)+\s", line)) for line in stream)
    return {"ptx_prefetch_instructions": ptx_count, "sass_decoded_instructions": sum(counts.values()),
            "sass_opcodes": dict(sorted(counts.items())), "cache_control_instructions": candidates,
            "disassembly_status": "decoded_opcode_observation" if counts else "unknown",
            "hint_retention": "unknown", "isa_reference": ISA_REFERENCE,
            "scope": "Cache-control opcodes are candidates, not proof of hint survival or removal."}


def validate(args):
    root = args.out.resolve()
    stages = args.steps.split()
    if not stages or len(set(stages)) != len(stages) or set(stages) - {"timed", "ncu", "sass", "analyze"}:
        raise ValueError("select explicit timed/ncu/sass/analyze stages")
    expected = requested_cells(args.cases, args.modes, args.ms)
    sources = {arm: (args.master_src if arm.startswith("base") or arm == "ncu-base" else args.new_src)
               / "tessera/serving/csrc/routed_fused_window.cu" for arm in (*ARMS, "ncu-base", "ncub-new")}
    receipt = {"schema": "tessera.f16_probe_scope.v1", "status": "selected_stages_passed",
               "stages": stages, "issue_acceptance": "not_established", "cells": 0,
               "scope": "Only selected synthetic probe stages; no full bitwise, dense timing, speed, NCU wait, or serving qualification."}
    if "timed" in stages or "analyze" in stages:
        documents = {arm: load(root / arm / "bench_pairs.json") for arm in ARMS}
        arms = {}
        for arm, doc in documents.items():
            native_identity(root, arm, doc, sources[arm])
            arms[arm] = cells(doc, expected, arm)
        rows = []
        for key in sorted(expected):
            if len({arms[arm][key]["out_sha256"] for arm in ARMS}) != 1:
                raise ValueError(f"output mismatch: {key}")
            row = dict(case=key[0], mode=key[1], M=key[2], bitwise=True)
            for arm in ARMS:
                row[f"{arm}_ms"] = arms[arm][key]["wall"]["median_ms"]
            for new, base in (("new", "base"), ("newb", "baseb")):
                row[f"{new}_over_{base}"] = row[f"{new}_ms"] / row[f"{base}_ms"]
            rows.append(row)
        (root / "ab_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        receipt["cells"] = len(rows)
    if "ncu" in stages:
        for arm in ("ncu-base", "ncub-new"):
            doc = load(root / arm / "bench_pairs.json")
            native_identity(root, arm, doc, sources[arm])
            cells(doc, requested_cells("r4", "0", "512"), arm, ncu=True)
            lines = (root / arm / "ncu.csv").read_text().splitlines()
            if not any("stall" in line.lower() or "scoreboard" in line.lower() for line in lines):
                raise ValueError(f"{arm}: NCU capture has no stall/scoreboard data")
    if "sass" in stages:
        receipt["sass"] = {arm: disassembly(None, root / f"sass-{arm}.txt") for arm in ("base", "new")}
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--master-src", type=Path)
    parser.add_argument("--new-src", type=Path)
    parser.add_argument("--steps", default="analyze")
    parser.add_argument("--cases", default="r4,r4+q,r3+q")
    parser.add_argument("--modes", default="0,2")
    parser.add_argument("--ms", default="1,512,2048")
    parser.add_argument("--native-path", choices=("base", "new"))
    parser.add_argument("--ptx", type=Path)
    parser.add_argument("--sass", type=Path)
    args = parser.parse_args()
    try:
        if args.sass is not None:
            print(json.dumps(disassembly(args.ptx, args.sass), indent=2))
        elif args.native_path:
            root = args.out.resolve()
            source_root = args.master_src if args.native_path == "base" else args.new_src
            source = source_root / "tessera/serving/csrc/routed_fused_window.cu"
            native = native_identity(root, args.native_path, load(root / args.native_path / "bench_pairs.json"), source)
            print(native["path"])
        else:
            receipt = validate(args)
            (args.out / "ab_acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"F16_PROBE_REFUSED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
