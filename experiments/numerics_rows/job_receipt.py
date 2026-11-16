"""Require a complete Q1 or H1 receipt after the container exits."""
import argparse
import json
import math
from pathlib import Path
import sys

from experiments.numerics_rows.common import Invalid, load_build, sha256, write_json


def require(condition, message):
    if not condition:
        raise Invalid(message)


def artifact(record):
    path = Path(record["path"])
    require(path.is_file() and sha256(path) == record["sha256"], f"missing or changed artifact: {path}")


def check(step, out, build_path):
    report = json.loads((out / f"{step.lower()}.json").read_text())
    registration = json.loads((out / "native_ops.json").read_text())
    require(registration == report["native_ops"] and registration["result"] == "PASS", "native ops not confirmed")
    scope = report["runtime"]["libraries"]
    built = load_build(build_path, report["runtime"])
    libraries = {r["key"]: r for r in built["libraries"]}
    require(scope == list(libraries), "runtime library scope differs from build")
    for record in libraries.values():
        for label in ("library", "probe"):
            artifact(record[label])
    proofs = report["tables"] if step == "Q1" else list(report["P3"].values())
    require(len(proofs) == len(libraries), "incomplete library checks")
    proofs = {r["key"]: r for r in proofs} if step == "Q1" else report["P3"]
    require(set(proofs) == set(libraries), "library check scope differs from build")
    for key, proof in proofs.items():
        for field, label in (("library_so_sha256", "library"), ("probe_sha256", "probe")):
            require(proof[field] == libraries[key][label]["sha256"], f"{key} {field} differs from build")
    if step == "H1":
        from experiments.numerics_rows.h1 import confirm_p3

        require(report["build_sha256"] == sha256(build_path), "H1 build receipt differs")
        require(report["preconditions_confirmed"] is True and report["gpu_preconditions_confirmed"] is True,
                "H1 is incomplete or failed")
        confirm_p3(report, proofs)
        require(report["preconditions_confirmed"] is True, "H1 proof is incomplete or failed")
    else:
        from experiments.numerics_rows.quantizers import families

        scope = families(libraries)
        require("e2m1" not in scope or report["artifact_scales"], "missing artifact quantizer scales")
        expected = {(family, gs) for family in scope
                    for gs in (set(report["artifact_scales"].values()) if family == "e2m1" else (1.0,))}
        actual = [(r["family"], r["gs"]) for r in report["quantizers"]]
        require(expected and set(actual) == expected and len(actual) == len(expected), "incomplete quantizer checks")
        require(all(math.isfinite(gs) and gs > 0 for _, gs in expected), "invalid quantizer scale")
        for record in report["quantizers"]:
            artifact(record)
            require(all(record[field] is True for field in ("repeat_equal", "finite_codes", "finite_scales")), "quantizer check failed")
            require(record["labels"], "missing quantizer input list")
        for table in proofs.values():
            size = table["dom_size"]
            require(type(size) is int and size > 0 and table["d_nondecreasing_0_L"] is True, "incomplete or decreasing D table")
            certificate = table["finite_certificate"]
            require(certificate["elements"] == size and certificate["finite"] is True
                    and type(certificate["magnitude_bounded"]) is bool
                    and certificate["held"] is certificate["magnitude_bounded"], "incomplete D certificate")
            for prefix in ("dom", "d_table"):
                artifact({"path": table[f"{prefix}_path"], "sha256": table[f"{prefix}_sha256"]})
                require(Path(table[f"{prefix}_path"]).stat().st_size == 4 * size, "incomplete D table bytes")
        require(len(report["boundaries"]) == 2 * len(expected)
                and all(r["kind"] == "quantizer" and r["sync_before"] is True and r["sync_after"] is True
                        and "message" not in r for r in report["boundaries"]), "incomplete quantizer boundaries")
    return {"verdict": "PASS", "step": step, "build_sha256": sha256(build_path),
            "receipt_sha256": sha256(out / f"{step.lower()}.json"), "libraries": sorted(libraries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("Q1", "H1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check(args.step, args.out, args.build)
    except Exception as error:
        result = {"verdict": "FAIL", "step": args.step, "error": {"type": type(error).__name__, "message": str(error)}}
    write_json(args.out / "receipt-check.json", result)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["verdict"] == "PASS" else 3)


if __name__ == "__main__":
    main()
