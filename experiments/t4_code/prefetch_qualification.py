"""TS875 finite qualification through the existing PB staged/native owners.

These are diagnostic kernel-bank calls, not production _ext serving admission.
No GPU action is authorized merely by the existence of this runner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/t8r_speed"), str(ROOT / "tests")]

SOURCE_SHA = "c2d56fc8b91045da3ff0b274e8cd7e285c1c5aa7b91bfc76fdc86886103e2012"
IMAGE = "vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
NATIVE_ROOT = Path("/mnt/shared/tessera-measurements/t4-875-cpu-20261003/native1-userenv")
NATIVE_RECORD = NATIVE_ROOT / "native_compile.json"
ARMS = {
    0: ("tessera_routed_fused_e2m1_dev_apf0", "9a33c215c306a75640d025b807e6e6d9aa7e75892d76598acdd2120360a5c26c"),
    4: ("tessera_routed_fused_e2m1_dev_apf4", "17f0f390ed688ba4a93876ec6713acf796550b64dba0f2e5d08f4d6e635fe3d9"),
}
PHASE = "native1-whole"
TOOLS_ROOT = NATIVE_ROOT.parent / "sanitizer-immutable"
TOOLS_SHA = {
    "compute-sanitizer": "7a7fcdefb67042731daf021478176f4919e1843d0b10cb697af28a7d8a3d108b",
    "libInterceptorInjectionTarget.so": "a8988c4ec7ba20d4b03bc27cbd0ad85dfe363a3a42c793e4a4883515c7a67ab0",
    "libsanitizer-collection.so": "468536e1b20e29241826788756f2b4a0f4c4e0cda8e6007c215164f590f23cf4",
    "libsanitizer-public.so": "3077e5732b363ff532e55e7e35c7ddff65d2c57d0898f0abc0b5d614a45d9a24",
    "libTreeLauncherPlaceholder.so": "7d5b878dc8d8a31c425d32e74cf473e2a0d5db6b8fc08829727c04d4caf5239c",
    "libTreeLauncherTargetInjection.so": "29862189027903d504b0d48d8f4583d36e38294d927371ef7405ef24155deac5",
    "libTreeLauncherTargetUpdatePreloadInjection.so": "e56384c6e944a9ef15d6a40d54f8cd4ebcb1d300a694bc44e8bb1d22408800e2",
    "TreeLauncherSubreaper": "9a27106d62ec6eb326abe2722e890d618d2e536fe5b9c9e9401d5b3369d7735f",
    "TreeLauncherTargetLdPreloadHelper": "9c3e6419b60038d39ab964f4fa4ed4c0a19d2db12540297fca2e2b136d39c811"
}



def validate_readset(path):
    from prismabuild import client, storage_tiers

    manifest, encoding = client.read_data_manifest(path)
    expected = [{"name": PHASE, "start_bytes": 0, "end_bytes": manifest["total_bytes"]}]
    if encoding != "identity" or storage_tiers.manifest_phase_ranges(manifest) != expected:
        raise ValueError("native1 whole-phase ranges differ or are incomplete")
    if client.manifest_read_entries(manifest) != manifest["entries"]:
        raise ValueError("native1 declared consumption order differs")
    required = {str(NATIVE_RECORD): (852, "b8b242cbbb2901320942e87a273cbce024e86276ad72ab66b23fe5778eb6d9d0")}
    for arm, (module, digest) in ARMS.items():
        required[str(NATIVE_ROOT / f"build_apf{arm}" / (module + ".so"))] = (1889616, digest)
    entries = {e["path"]: e for e in manifest["entries"]}
    for name, (size, digest) in required.items():
        entry = entries.get(name)
        if entry is None or (entry["offset"], entry["bytes"], entry["sha256"]) != (0, size, digest):
            raise ValueError(f"native1 declared artifact differs: {name}")
    native_names = set(required)
    tool_names = {str(TOOLS_ROOT / name) for name in TOOLS_SHA}
    if set(entries) not in (native_names, native_names | tool_names):
        raise ValueError("unqualified native1 readset members")
    for name, digest in TOOLS_SHA.items():
        entry = entries.get(str(TOOLS_ROOT / name))
        if entry is not None and (entry["offset"], entry["sha256"]) != (0, digest):
            raise ValueError(f"staged sanitizer artifact differs: {name}")
    return manifest, expected


def repair_readset(original, output, *, include_tools=False):
    value = json.loads(Path(original).read_bytes())
    original_entries = list(value["entries"])
    if include_tools:
        tools = []
        for name, expected in TOOLS_SHA.items():
            path = TOOLS_ROOT / name
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != expected:
                raise ValueError(f"tool preparation bytes differ: {path}")
            tools.append({"path": str(path), "offset": 0, "bytes": len(raw), "sha256": digest})
        value["entries"] = tools + original_entries
        value["entry_count"] = len(value["entries"])
        value["total_bytes"] = sum(e["bytes"] for e in value["entries"])
    value["annotations"] = dict(value["annotations"], phases=[
        {"name": PHASE, "bytes": value["total_bytes"], "cumulative_bytes": value["total_bytes"]}])
    with Path(output).open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    checked, phases = validate_readset(output)
    if [e for e in checked["entries"] if e["path"] in {x["path"] for x in original_entries}] != original_entries:
        raise ValueError("readset repair changed retained input bytes")
    print(json.dumps({"readset": str(output), "sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest(),
                      "entry_count": checked["entry_count"], "total_bytes": checked["total_bytes"],
                      "phases": phases, "retained_entries_unchanged": True}), flush=True)


def staged_read(path):
    from pb_staged_store import StagedInputs

    manifest, phases = validate_readset(path)
    reader = StagedInputs(path)
    try:
        for entry in manifest["entries"]:
            reader.read(entry["path"], entry["offset"])
        if sum(row["bytes"] for row in reader.reads) != manifest["total_bytes"]:
            raise ValueError("native1 staged-reader coverage differs")
        print(json.dumps({"kind": "CPU staged-reader proof only", "action_key": reader.ctx["action_key"],
                          "pin_id": reader.held["pin_id"], "ref_id": reader.held["ref_id"],
                          "readset_sha256": reader.manifest_sha256, "phases": phases,
                          "reads": reader.reads}), flush=True)
    finally:
        reader.close()


@contextmanager
def leased_banks(manifest_path, out, *, gpu):
    import torch
    from pb_staged_store import NativeCallback, StagedInputs
    from tessera import routed_fused_e2m1 as fe
    from tessera.serving import ext

    validate_readset(manifest_path)
    if gpu and (not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1)):
        raise ValueError("the diagnostic FP4 kernel matrix requires an admitted GB10 GPU")
    if not gpu and torch.cuda.is_available():
        raise ValueError("CPU native-map proof must have GPU visibility disabled")
    reader = StagedInputs(manifest_path)
    owners, banks = [], {}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fence = torch.cuda.synchronize if gpu else lambda: None
    try:
        record = reader.json(NATIVE_RECORD)
        if (record["source_sha256"], record["torch"], record["image"]) != (
                SOURCE_SHA, torch.__version__, os.environ.get("ORACLE_IMAGE")):
            raise ValueError("native1 source/torch/image identity differs")
        rows = {row["activation_prefetch"]: row for row in record["banks"]}
        if set(rows) != set(ARMS):
            raise ValueError("native1 must name exactly the baseline/candidate arms")
        source = ext.native_source_path(fe.MODULE_NAME_VALUE)
        for arm, (module, digest) in ARMS.items():
            row = rows[arm]
            expected_path = NATIVE_ROOT / f"build_apf{arm}" / (module + ".so")
            if (row["path"], row["sha256"], row["bytes"]) != (str(expected_path), digest, 1889616):
                raise ValueError("native1 module/PyInit artifact differs")
            owner = NativeCallback(reader, expected_path, fe, out / f"arm{arm}",
                                   expected_sha256=digest, source_sha256=SOURCE_SHA,
                                   module=module, source_module=fe.MODULE_NAME_VALUE,
                                   install_build_callback=False)
            owners.append(owner)
            bank = owner.load_declared(source)
            fe._check_library(bank, arm)
            owner.attest_mapped(bank)
            banks[arm] = bank
        yield banks
    finally:
        # Both original load-FD names stay distinct and alive through every
        # call/graph/reduction. No nested build callbacks or per-case reloads.
        try:
            fence()
            for owner in owners:
                if owner.module is not None:
                    owner.attest_mapped(owner.module)
                owner.finish(fence, keep_load_fd=True)
                print("T4_STAGED_NATIVE_IDENTITY " + json.dumps(owner.record), flush=True)
            (out / "native-identities.json").write_text(json.dumps([o.record for o in owners], indent=2))
            print(json.dumps({"readset_sha256": reader.manifest_sha256,
                              "pin_id": reader.held["pin_id"], "ref_id": reader.held["ref_id"],
                              "kind": "diagnostic code banks; not serving _ext admission"}), flush=True)
        finally:
            if any(fe._LIB is bank for bank in banks.values()):
                fe._LIB, fe._LIB_PREFETCH = None, None
            try:
                for owner in owners:
                    if owner.closed and owner.module is not None:
                        os.close(owner.fd)
                    elif not owner.closed:
                        owner.finish(fence)
            finally:
                reader.close()


class NativePopulation:
    """Frozen diagnostic scope: 42 FP4 native cases plus six terminal patterns."""
    def __init__(self):
        self.expected, self.passed, self.skipped = set(), set(), set()

    def pytest_collection_finish(self, session):
        reasons = {"the block-scaled FP4 instruction is sm_121a", "native terminal CUDA coverage"}
        self.expected = {item.nodeid for item in session.items
                         if any(mark.kwargs.get("reason") in reasons for mark in item.iter_markers("skipif"))}
        if len(self.expected) != 48:
            raise ValueError("the frozen 48-case native diagnostic population changed")

    def pytest_runtest_logreport(self, report):
        if report.nodeid in self.expected:
            if report.skipped:
                self.skipped.add(report.nodeid)
            if report.when == "call" and report.passed:
                self.passed.add(report.nodeid)

    def require_complete(self):
        if self.skipped or self.passed != self.expected:
            raise ValueError("incomplete native diagnostic execution; skips are not qualification")
        print(json.dumps({"native_expected": len(self.expected), "native_passed": len(self.passed),
                          "native_skipped": len(self.skipped)}), flush=True)


def consume(args):
    with leased_banks(args.manifest, args.out, gpu=not args.cpu_map) as banks:
        if args.cpu_map:
            print(json.dumps({"kind": "CPU held staged native-map proof", "arms": list(banks)}), flush=True)
            return 0
        if args.arm is None:
            from fused_e2m1_check import main
            return main(["--out", str(Path(args.out) / "numeric"), "--compare-prefetch"], banks=banks)
        import pytest
        from tessera import routed_fused_e2m1 as fe
        os.environ[fe.PREFETCH_ENV] = str(args.arm)
        fe._LIB, fe._LIB_PREFETCH = banks[args.arm], args.arm
        if fe._ext() is not banks[args.arm]:
            raise ValueError("native consumer did not preserve its exact selected bank")
        population = NativePopulation()
        code = pytest.main(["-q", "-ra", "tests/test_routed_fused_e2m1.py",
                            "tests/test_routed_terminal_cuda.py", "-k", "e2m1"], plugins=[population])
        if code == 0:
            population.require_complete()
        return code


def sanitize(args):
    from pb_staged_store import StagedInputs

    validate_readset(args.manifest)
    reader = StagedInputs(args.manifest)
    out = Path(args.out)
    tool_dir = out / "staged-sanitizer"
    tool_dir.mkdir(parents=True, exist_ok=False)
    try:
        for name, expected in TOOLS_SHA.items():
            target = tool_dir / name
            target.write_bytes(reader.read(TOOLS_ROOT / name))
            target.chmod(0o755)
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError("actual staged sanitizer bytes differ")
        if args.command == "tool-preflight":
            subprocess.run([str(tool_dir / "compute-sanitizer"), "--version"], check=True)
            print(json.dumps({"kind": "CPU staged-tool compatibility only",
                              "pin_id": reader.held["pin_id"], "ref_id": reader.held["ref_id"],
                              "reads": reader.reads}), flush=True)
            return 0
        command = [str(tool_dir / "compute-sanitizer"), "--tool", args.tool,
                   "--error-exitcode", "86", "--target-processes", "all",
                   "--log-file", str(out / "sanitizer.log"), sys.executable, str(Path(__file__).resolve()),
                   "consume", "--manifest", args.manifest, "--out", str(out / "checked")]
        if args.arm is not None:
            command.extend(["--arm", str(args.arm)])
        subprocess.run(command, check=True)
        return 0
    finally:
        reader.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    repair = sub.add_parser("repair-readset")
    repair.add_argument("--original", required=True)
    repair.add_argument("--out", required=True)
    repair.add_argument("--include-tools", action="store_true")
    validate = sub.add_parser("validate-readset")
    validate.add_argument("--manifest", required=True)
    read = sub.add_parser("staged-read")
    read.add_argument("--manifest", required=True)
    run = sub.add_parser("consume")
    run.add_argument("--manifest", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--arm", type=int, choices=(0, 4))
    run.add_argument("--cpu-map", action="store_true")
    san = sub.add_parser("sanitize")
    san.add_argument("--manifest", required=True)
    san.add_argument("--out", required=True)
    san.add_argument("--tool", required=True, choices=("memcheck", "racecheck", "synccheck", "initcheck"))
    san.add_argument("--arm", type=int, choices=(0, 4))
    tools = sub.add_parser("tool-preflight")
    tools.add_argument("--manifest", required=True)
    tools.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.command == "repair-readset":
        repair_readset(args.original, args.out, include_tools=args.include_tools)
        return 0
    if args.command == "validate-readset":
        manifest, phases = validate_readset(args.manifest)
        print(json.dumps({"entry_count": manifest["entry_count"], "total_bytes": manifest["total_bytes"],
                          "phases": phases}), flush=True)
        return 0
    if args.command == "staged-read":
        staged_read(args.manifest)
        return 0
    if args.command in ("sanitize", "tool-preflight"):
        return sanitize(args)
    return consume(args)


if __name__ == "__main__":
    raise SystemExit(main())
