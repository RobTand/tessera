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

SOURCE_SHA = "ebdfa49e0f8d3389d0e0be59821546668a08a81bea104d47f1650e34e63465d3"
NATIVE_ROOT = Path("/mnt/shared/tessera-measurements/t4-875-composed-qual-20261004/cpu-builds-a3a599ec")
BANKS = {
    0: {"module": "tessera_routed_fused_e2m1", "bytes": 1892000,
        "sha256": "1302ad713309ef54a62a6a4ff9edeed7d61306803ce820df9c704f3ca279afbf",
        "record_bytes": 2425, "record_sha256": "8545d98c733ff884933fe50018190023632f13c8ee77e655f839c39e862ad0b4",
        "finalization_bytes": 2329, "finalization_sha256": "5a243e4241e8f5f11b9a395381bb64c57d50626d3c5f0dcafae03a54ccb3a7b6",
        "action_key": "917cc5a474d39d0b97db9fbeb65939dba68dac62c11ad5ff08db340ec5bbf0f7"},
    4: {"module": "tessera_routed_fused_e2m1_apf4", "bytes": 1892040,
        "sha256": "7111322823e7677795e7e18df11bdff5d0e233f0672383125950f950c1cc2c1a",
        "record_bytes": 2445, "record_sha256": "1685d289ff59fae431cee4f1061977ce4182f194c1d91797a7981c99123ceef9",
        "finalization_bytes": 2369, "finalization_sha256": "4e44e45931ea337cf9f16cfc38a69a4d802ef9a257bdbf5f1f7fa787bd28f81c",
        "action_key": "b9e2173f70f1bba64b0142454ababd45ad36bb435fc6be12237f2bff1e674b83"},
}
for _arm, _bank in BANKS.items():
    _bank["root"] = NATIVE_ROOT / f"ext-fp4-b{_arm}"
    _bank["path"] = _bank["root"] / (_bank["module"] + "_sm_121_tessera_guarded_v1") / (_bank["module"] + ".so")
    _bank["record_path"] = _bank["root"] / "native-build-record.json"
    _bank["finalization_path"] = _bank["path"].parent / "native-finalization.json"
PHASE = "t4-composed-whole"
TOOLS_ROOT = Path("/mnt/shared/tessera-measurements/t4-875-cpu-20261003/sanitizer-immutable")
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



def native_members():
    """Exact production-bank record/finalization/ELF order, one existing owner."""
    members = {}
    for bank in BANKS.values():
        for name in ("record", "finalization"):
            members[str(bank[name + "_path"])] = (bank[name + "_bytes"], bank[name + "_sha256"])
        members[str(bank["path"])] = (bank["bytes"], bank["sha256"])
    return members


def validate_bank_record(record, finalization, arm, *, source_module):
    """Bind the actual production-bank compile/finalization, not old proofs."""
    bank = BANKS[arm]
    if (record.get("schema"), record.get("action_key"), record.get("source_sha256")) != (
            "tessera.native_build_cohort.v1", bank["action_key"], SOURCE_SHA):
        raise ValueError("common native compile source/action identity differs")
    image = record["image"]
    from tessera.serving.contract import require_runtime_image
    reference = require_runtime_image(image["required"], "qualified native compile image")
    if (image.get("schema"), image.get("required"), image.get("requested"), image.get("resolved_reference"),
            image.get("present"), image.get("refused")) != (
            "tessera.runtime_image/1", reference, reference, reference, True, False):
        raise ValueError("common native compile image identity differs")
    selectors = {"TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH": "0",
                 "TESSERA_ROUTED_FUSED_FP4_A_PREFETCH": str(arm),
                 "TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH": "0"}
    if record.get("selectors") != selectors:
        raise ValueError("common native compile independent selectors differ")
    expected = {"library": "e2m1", "module": bank["module"], "source_module": source_module,
                "path": str(bank["path"]), "bytes": bank["bytes"], "sha256": bank["sha256"],
                "finalization_path": str(bank["finalization_path"]),
                "finalization_sha256": bank["finalization_sha256"],
                "status": "compile gate (no matching device here)"}
    if record.get("libraries") != [expected]:
        raise ValueError("common native production module/artifact differs")
    if (finalization.get("schema"), finalization.get("reconciled"), finalization.get("no_pending_work")) != (
            "tessera.native_build_finalization.v1", True, True):
        raise ValueError("native build is not finalized with no pending work")
    bindings = finalization["bindings"]
    if bindings[record["source_path"]]["sha256"] != SOURCE_SHA:
        raise ValueError("finalization source binding differs")
    if (bindings[str(bank["path"])]["sha256"], bindings[str(bank["path"])]["bytes"]) != (bank["sha256"], bank["bytes"]):
        raise ValueError("finalization ELF binding differs")
    return reference


def validate_readset(path):
    from prismabuild import client, storage_tiers

    manifest, encoding = client.read_data_manifest(path)
    expected = [{"name": PHASE, "start_bytes": 0, "end_bytes": manifest["total_bytes"]}]
    if encoding != "identity" or storage_tiers.manifest_phase_ranges(manifest) != expected:
        raise ValueError("native1 whole-phase ranges differ or are incomplete")
    if client.manifest_read_entries(manifest) != manifest["entries"]:
        raise ValueError("native1 declared consumption order differs")
    required = native_members()
    ordered = [e["path"] for e in manifest["entries"]]
    if any(e["offset"] != 0 for e in manifest["entries"]) or len(set(ordered)) != len(ordered):
        raise ValueError("native1 needs unique whole-file offset-zero members")
    native_order = list(required)
    tool_order = [str(TOOLS_ROOT / name) for name in TOOLS_SHA]
    if ordered not in (native_order, tool_order + native_order):
        raise ValueError("native1 exact normalized member order differs")
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


def prepare_readset(output, *, include_tools=False):
    """Seal actual already-finalized inputs; never compile or substitute banks."""
    members = native_members()
    if include_tools:
        members = {**{str(TOOLS_ROOT / name): (None, digest) for name, digest in TOOLS_SHA.items()}, **members}
    entries = []
    for name, (expected_size, expected_digest) in members.items():
        raw = Path(name).read_bytes()
        if (expected_size is not None and len(raw) != expected_size) or hashlib.sha256(raw).hexdigest() != expected_digest:
            raise ValueError(f"qualified artifact changed before sealing: {name}")
        entries.append({"path": name, "offset": 0, "bytes": len(raw), "sha256": expected_digest})
    total = sum(entry["bytes"] for entry in entries)
    value = {"schema": "prismaquant.prismabuild.data_manifest.v1",
             "produced_by": {"issue": 875, "scope": "actual finalized common-source T4 code banks; not GPU proof"},
             "mount_prefix": "/mnt/shared", "entries": entries, "entry_count": len(entries), "total_bytes": total,
             "annotations": {"phases": [{"name": PHASE, "bytes": total, "cumulative_bytes": total}]}}
    with Path(output).open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    validate_readset(output)
    print(json.dumps({"readset": str(output), "sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest(),
                      "entries": len(entries), "bytes": total, "scope": "sealed metadata, no CUDA execution"}), flush=True)


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


def finish_banks(reader, owners, fence, *, primary=None):
    """Attempt every held-artifact closure before releasing the SDK reader."""
    errors = []
    for owner in owners:
        loaded = owner.module is not None
        try:
            if loaded:
                owner.attest_mapped(owner.module)
        except BaseException as error:
            errors.append(error)
        try:
            owner.finish(fence, keep_load_fd=True)
        except BaseException as error:
            errors.append(error)
        finally:
            if loaded or not owner.closed:
                try:
                    os.close(owner.fd)
                    owner.closed = True
                except BaseException as error:
                    errors.append(error)
    try:
        reader.close()
    except BaseException as error:
        errors.append(error)
    if errors:
        if primary is not None:
            for error in errors:
                primary.add_note(f"staged cleanup failure: {error!r}")
        elif len(errors) == 1:
            raise errors[0]
        else:
            raise BaseExceptionGroup("staged native teardown failures", errors)


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
    fence = torch.cuda.synchronize if gpu else lambda: None
    try:
        out.mkdir(parents=True, exist_ok=True)
        source = ext.native_source_path(fe.MODULE_NAME_VALUE)
        for arm, expected_bank in BANKS.items():
            record = reader.json(expected_bank["record_path"])
            finalization = reader.json(expected_bank["finalization_path"])
            reference = validate_bank_record(record, finalization, arm, source_module=fe.MODULE_NAME_VALUE)
            if os.environ.get("ORACLE_IMAGE") != reference:
                raise ValueError("the T4 consumer must use its exact recorded common compile image")
            owner = NativeCallback(reader, expected_bank["path"], fe, out / f"arm{arm}",
                                   expected_sha256=expected_bank["sha256"], source_sha256=SOURCE_SHA,
                                   module=expected_bank["module"], source_module=fe.MODULE_NAME_VALUE,
                                   install_build_callback=False)
            owners.append(owner)
            bank = owner.load_declared(source)
            fe._check_library(bank, arm)
            owner.attest_mapped(bank)
            banks[arm] = bank
        yield banks
    finally:
        primary = sys.exc_info()[1]
        if any(fe._LIB is bank for bank in banks.values()):
            fe._LIB, fe._LIB_PREFETCH = None, None
        # The shared owner fences each mapped bank; cleanup never stops at the
        # first failed bind/hash/fence and always releases after every FD attempt.
        finish_banks(reader, owners, fence, primary=primary)
        if primary is None:
            (out / "native-identities.json").write_text(json.dumps([o.record for o in owners], indent=2))
            for owner in owners:
                print("T4_STAGED_NATIVE_IDENTITY " + json.dumps(owner.record), flush=True)
            print(json.dumps({"readset_sha256": reader.manifest_sha256,
                              "pin_id": reader.held["pin_id"], "ref_id": reader.held["ref_id"],
                              "kind": "diagnostic code banks; not serving _ext admission"}), flush=True)


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
    try:
        tool_dir.mkdir(parents=True, exist_ok=False)
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
    prepare = sub.add_parser("prepare-readset")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--include-tools", action="store_true")
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
    if args.command == "prepare-readset":
        prepare_readset(args.out, include_tools=args.include_tools)
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
