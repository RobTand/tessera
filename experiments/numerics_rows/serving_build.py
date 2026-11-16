"""Bind the serving loader's source path to a retained P build."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import json
from pathlib import Path
import sys

from experiments.numerics_rows.common import Invalid, KERNEL, KERNEL_SHA256, check_runtime, check_source, sha256, write_json


def source_provenance(build_path, out):
    """Copy P metadata and add the literal compile input from its build.ninja."""
    built = json.loads(Path(build_path).read_text())
    for record in built["libraries"]:
        ninja = Path(record["library"]["path"]).parent / "build.ninja"
        source, = [line.split(": cuda_compile ", 1)[1] for line in ninja.read_text().splitlines() if ": cuda_compile " in line]
        if not Path(source).is_absolute() or "$" in source or not source.endswith("/" + KERNEL) or sha256(source) != KERNEL_SHA256:
            raise Invalid("P build has no literal, pinned kernel compile input")
        record.update(native_source_path=source, native_source_sha256=KERNEL_SHA256,
                      build_ninja={"path": str(ninja), "sha256": sha256(ninja)})
    built.update(build_path=str(build_path), build_sha256=sha256(build_path))
    write_json(out, built)
    return built


def source_layout(build_path, provenance_path):
    """Require an unchanged P copy with one absolute source root for the mount."""
    built = json.loads(Path(build_path).read_text())
    proof = json.loads(Path(provenance_path).read_text())
    if proof.get("build_sha256") != sha256(build_path) or len(proof["libraries"]) != len(built["libraries"]):
        raise Invalid("P source provenance belongs to another build")
    for original, record in zip(built["libraries"], proof["libraries"]):
        if any(record.get(k) != v for k, v in original.items()):
            raise Invalid("P source provenance changed retained build metadata")
        path = record.get("native_source_path", "")
        if not Path(path).is_absolute() or not path.endswith("/" + KERNEL) or record.get("native_source_sha256") != KERNEL_SHA256:
            raise Invalid("P source provenance has no pinned absolute compile path")
    roots = {str(Path(r["native_source_path"]).parents[4]) for r in proof["libraries"]}
    if len(roots) != 1:
        raise Invalid("P libraries do not share one source mount")
    return Path(roots.pop()), proof


def check_compile_paths(build_path, provenance_path):
    from tessera.serving.ext import native_source_path
    from tessera.routed_fused import _cflags

    _, proof = source_layout(build_path, provenance_path)
    records = []
    for record in proof["libraries"]:
        actual = native_source_path(record["name"])
        key = record["key"]
        flags = _cflags("sm_121", key in ("e4m3", "e4m3mma"), key == "e4m3mma", key == "e2m1")
        if actual != record["native_source_path"] or sha256(actual) != record["native_source_sha256"] or flags != record["cflags"]:
            raise Invalid(f"serving loader compile path, source or flags differ from P: expected {record['native_source_path']}, observed {actual}")
        records.append({"key": key, "native_source_path": actual, "native_source_sha256": sha256(actual), "cflags": flags})
    return records


def compare_binaries(measured, served, measured_sass, out):
    from experiments.numerics_rows.canonical import CANONICAL_FIELDS, explain, mask, negative_controls

    out.mkdir(parents=True)
    witness = {"measured": Path(measured_sass), "served": Path(served["sass"]["path"])}
    sass = {name: mask(path.read_bytes())[0] for name, path in witness.items()}
    sass_equal = sass["measured"] == sass["served"]
    audit = explain(measured["snapshot_path"], served["snapshot_path"],
                    measured["library_canonical_sections"], served["library_canonical_sections"])
    device_audits = [explain(a["path"], b["path"], a["sections"], b["sections"])
                     for a, b in zip(measured["canonical_cubins"], served["canonical_cubins"])]
    cubin_paths = [Path(r["path"]) for r in measured["canonical_cubins"]]
    controls = negative_controls(Path(measured["snapshot_path"]).read_bytes(), [p.read_bytes() for p in cubin_paths], [p.name for p in cubin_paths])
    # cuobjdump -ptx must find no additional, unbound device program.
    ptx = {}
    for name, binary in (("measured", measured["snapshot_path"]), ("served", served["snapshot_path"])):
        output = out / f"{name}.ptx"
        with output.open("w") as stream:
            subprocess.run(["cuobjdump", "-ptx", binary], stdout=stream, check=True)
        ptx[name] = {"path": str(output), "sha256": sha256(output), "empty": not output.read_text().strip()}
    compressed = all(b"\ncompressed\n" in path.read_bytes() for path in witness.values())
    equal = (all(served[field] == measured[field] for field in CANONICAL_FIELDS) and sass_equal
             and audit["all_differences_explained"] and len(device_audits) == len(measured["canonical_cubins"]) == len(served["canonical_cubins"])
             and all(r["all_differences_explained"] for r in device_audits) and all(r["verdict"] == "DIFFERENT" for r in controls.values())
             and compressed and all(r["empty"] for r in ptx.values()))
    return {"identity_equal": equal, "measured": measured, "served": served,
            "raw_file_audit": audit, "cubin_audits": device_audits, "negative_controls": controls,
            "sass_token_masked_equal": sass_equal,
            "sass_witnesses": {k: {"path": str(witness[k]), "sha256": hashlib.sha256(v).hexdigest()} for k, v in sass.items()},
            "compressed_fatbin": compressed, "ptx": ptx}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "build", "provenance", "h1", "population", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--retained-loader-proof", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True)
    check_source(args.source)
    runtime = check_runtime("N")
    sys.path.insert(0, str(args.source / "src"))
    import torch
    from torch.utils import cpp_extension
    from tessera import routed_fused as rf
    from tessera.serving.backend import PlatformMismatchError
    from experiments.numerics_rows.row import unit_libraries
    from experiments.numerics_rows.build import IDENTITY_FIELDS, library_identity
    from experiments.numerics_rows.population import load_signed_population

    torch.set_num_threads(1)
    _, population_digest = load_signed_population(args.population)
    if torch.cuda.is_available():
        raise Invalid("this is a CPU-only serving-loader compile proof")
    paths = check_compile_paths(args.build, args.provenance)
    h1 = json.loads(args.h1.read_text())
    if not h1["preconditions_confirmed"] or h1["build_sha256"] != sha256(args.build):
        raise Invalid("proof requires H1 for this retained P build")
    report = {"runtime": runtime, "compile_paths": paths, "build_sha256": sha256(args.build),
              "population_sha256": population_digest,
              "h1_sha256": sha256(args.h1), "cpu_only": True, "libraries": [], "verdict": "INCOMPLETE"}
    built = json.loads(args.build.read_text())
    enriched_h1 = json.loads(args.h1.read_text())
    enriched_h1["canonical_source_h1_sha256"] = sha256(args.h1)
    original_load = cpp_extension.load
    captured = []
    def capture(**kwargs):
        lib = original_load(**kwargs)
        captured.append(lib)
        return lib
    cpp_extension.load = capture
    cache = {}
    for path in paths:
        key = path["key"]
        rf._ext.cache_clear()
        captured.clear()
        try:
            rf._ext(key)
        except PlatformMismatchError as error:
            refusal = str(error)
        else:
            raise Invalid("CPU compile proof unexpectedly admitted a serving platform")
        lib, = captured
        served, = unit_libraries(lib, Path(path["native_source_path"]), args.out / "binaries", cache)
        record, = [r for r in built["libraries"] if r["key"] == key]
        measured = library_identity(record["library"]["path"], args.out / "measured" / key / "cubins")
        if any(measured[k] != h1["P3"][key].get(k) for k in IDENTITY_FIELDS):
            raise Invalid(f"retained P raw identity differs from H1 for {key}")
        comparison = compare_binaries(measured, served, record["library"]["sass"]["path"], args.out / "comparison" / key)
        if args.retained_loader_proof and key == "e4m3mma":
            prior = json.loads(args.retained_loader_proof.read_text())
            retained, = [r for r in prior["libraries"] if r["key"] == key]
            old = library_identity(retained["snapshot_path"], args.out / "retained-loader" / key / "cubins")
            if old["library_so_sha256"] != retained["library_so_sha256"]:
                raise Invalid("retained failed loader snapshot changed")
            old["sass"] = retained["sass"]
            comparison["retained_failed_loader"] = compare_binaries(measured, old, record["library"]["sass"]["path"], args.out / "retained-comparison" / key)
            comparison["identity_equal"] &= comparison["retained_failed_loader"]["identity_equal"]
        report["libraries"].append({"key": key, "uses": ["dense", "routed"] if key == "e4m3mma" else ["dense"],
                                    "platform_refusal": refusal, **comparison})
        record.update(measured)
        enriched_h1["P3"][key].update({k: v for k, v in measured.items() if k not in ("library_so_sha256", "library_code_sha256", "library_code_sections", "cubin_sha256s")})
        write_json(args.out / "proof.json", report)
    report["verdict"] = "SAME" if all(r["identity_equal"] for r in report["libraries"]) else "DIFFERENT"
    write_json(args.out / "proof.json", report)
    write_json(args.out / "p-canonical.json", {**built, "build_sha256": sha256(args.build)})
    write_json(args.out / "h1-canonical.json", enriched_h1)
    if report["verdict"] != "SAME":
        raise Invalid("canonical identity is not established; every unexplained byte remains DIFFERENT")



if __name__ == "__main__":
    main()
