"""Bind loaded device code, not source names or compiler cache keys.

The Triton adapter targets IMG-N's 3.7.1 CompiledKernel API. It retains the
bytes passed to _init_handles and checks them before the run callable. Existing
launch hooks remain unchanged. Unknown APIs or missing evidence cannot attest.
Image origins require an exact CUDA function name in a hashed image file,
loaded at its original image path from the same executable-mapped inode.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace

from experiments.numerics_rows.common import Invalid, Unattested, sha256
from experiments.numerics_rows.launch_state import json_digest

IMAGE_CODE = Path(__file__).with_name("dependency-image.json")


def canonical(value):
    """Keep types and float bits, including negative zero, in specialization data."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return {"float_hex": value.hex()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return sorted([[canonical(k), canonical(v)] for k, v in value.items()], key=json_digest)
    if is_dataclass(value):
        return canonical(asdict(value))
    if hasattr(value, "_asdict"):
        return canonical(value._asdict())
    raise Unattested(f"unsupported Triton specialization value: {type(value).__name__}")


def definition(kernel):
    target = kernel.metadata.target
    if target.backend != "cuda":
        raise Unattested("Triton specialization target is not CUDA")
    return {"entry": kernel.name, "target": {k: canonical(getattr(target, k)) for k in ("backend", "arch", "warp_size")},
            "constants": canonical(kernel.src.constants), "metadata": canonical(kernel.metadata._asdict()),
            "packed_metadata": canonical(kernel.packed_metadata)}


def specialization(kernel, launch, loaded):
    if loaded is None:
        raise Unattested("Triton specialization was loaded before the code observer")
    data, function, bound = loaded
    if kernel.kernel != data or kernel.asm.get("cubin") != data or kernel.function != function:
        raise Invalid("loaded Triton cubin or function changed")
    if len(launch) < 6 or launch[4] != function or launch[5] != kernel.packed_metadata:
        raise Invalid("Triton launch does not use the bound function or parameters")
    if definition(kernel) != bound:
        raise Invalid("Triton loaded specialization metadata changed")
    return {"cubin_sha256": hashlib.sha256(data).hexdigest(), **{k: bound[k] for k in ("entry", "target", "constants")},
            "launch": {"grid": canonical(launch[:3]), **{k: bound[k] for k in ("metadata", "packed_metadata")}}}


def binding(unit_id, case, identity):
    record = {"unit_id": unit_id, "case": case, "identity": identity}
    return {**record, "sha256": json_digest(record)}


def image_symbols(path):
    """cuobjdump's ELF function roster avoids disassembling whole torch libraries."""
    result = subprocess.run(["cuobjdump", "-symbols", str(path)], capture_output=True, text=True, errors="replace")
    if result.returncode:
        cause = (result.stderr.splitlines() or [""])[0][:200]
        raise Unattested(f"image CUDA symbol roster unavailable: {path}: cuobjdump -symbols exit {result.returncode}: {cause}")
    # ELF entries printed by cuobjdump: STT_FUNC <binding> <visibility> [U] <exact mangled name>; the type is the first field.
    rows = [line.split() for line in result.stdout.splitlines()]
    functions = sorted({fields[-1] for fields in rows if fields[:1] == ["STT_FUNC"] and len(fields) >= 4})
    names = subprocess.check_output(["c++filt"], input="\n".join(functions) + "\n", text=True).splitlines() if functions else []
    return [{"symbol": symbol, "demangled": name} for symbol, name in zip(functions, names, strict=True)]


ROSTER_REFUSAL = "mapped file or device-code roster unavailable: "
EMPTY_ROSTERS = Path(__file__).with_name("empty-rosters.json")


# Linux memfd and seal constants (linux/memfd.h, linux/fcntl.h); some Python builds do not export them.
MFD_CLOEXEC, MFD_ALLOW_SEALING, F_ADD_SEALS = 0x1, 0x2, 1033
F_SEAL_SEAL, F_SEAL_SHRINK, F_SEAL_GROW, F_SEAL_WRITE = 0x1, 0x2, 0x4, 0x8


def sealable_memfd():
    import ctypes, os
    if hasattr(os, "memfd_create"):
        return os.memfd_create("roster-snapshot", MFD_CLOEXEC | MFD_ALLOW_SEALING)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.memfd_create(b"roster-snapshot", MFD_CLOEXEC | MFD_ALLOW_SEALING)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create failed")
    return fd


def proved_empty_roster(path, data, live):
    """An empty roster counts only for a file whose verified bytes are a reviewed entry of empty-rosters.json.

    The roster is listed again from a sealed in-memory snapshot of exactly those bytes, and the live descriptor `live`
    must still hold them afterwards, so the hash, the empty listing and the mapped inode are the same bytes."""
    import fcntl, os
    digest = hashlib.sha256(data).hexdigest()
    evidence = EMPTY_ROSTERS.read_bytes()
    entry = json.loads(evidence)["files"].get(digest)
    if entry is None:
        raise Unattested(f"mapped device-code roster is empty and sha256 {digest} has no reviewed proof")
    snapshot = sealable_memfd()
    try:
        with os.fdopen(snapshot, "wb", closefd=False) as stream:
            stream.write(data)
        fcntl.fcntl(snapshot, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL)
        if image_symbols(f"/proc/{os.getpid()}/fd/{snapshot}"):
            raise Unattested(f"mapped device-code roster changed while it was read: sha256 {digest}")
    finally:
        os.close(snapshot)
    os.lseek(live, 0, os.SEEK_SET)
    held = b""
    while len(held) <= len(data) and (chunk := os.read(live, len(data) + 1 - len(held))):
        held += chunk
    if hashlib.sha256(held).hexdigest() != digest:
        raise Unattested(f"mapped device-code roster changed while it was read: sha256 {digest}")
    return {"path": str(path), "sha256": digest, "elf_count": entry["elf"], "ptx_count": entry["ptx"],
            "evidence_sha256": hashlib.sha256(evidence).hexdigest(),
            **{f"{k}_sha256": v for k, v in entry["listings_sha256"].items()}}


def mapped_executables(maps=None):
    """Each executable mapping with its file's size and change time; a write to the same inode changes the entry."""
    import os
    def stat(path):
        try:
            s = os.stat(path)
            return [s.st_size, s.st_ctime_ns, s.st_ino, s.st_dev]
        except OSError:
            return None
    lines = (Path("/proc/self/maps").read_text() if maps is None else maps).splitlines()
    return sorted({json.dumps([f[5], f[3], f[4], stat(f[5])]) for f in map(lambda l: l.split(None, 5), lines) if len(f) == 6 and "x" in f[1]})


def origin_index(libraries=(), specializations=()):
    normalize = lambda s: re.sub(r"\s+", "", s)
    index = {}
    def add(name, record):
        index.setdefault(normalize(name), {})[json_digest(record)] = record
    for library in libraries:
        for symbol in library["symbols"]:
            record = {"origin": library["origin"], "path": library["path"],
                      "sha256": library["sha256"], "symbol": symbol["symbol"]}
            for key in ("symbol", "demangled"):
                add(symbol[key], record)
    for record in specializations:
        identity = record["identity"]
        add(identity["entry"], {"origin": "compiled", "code": {k: v for k, v in identity.items() if k != "launch"}})
    return index


def kernel_origins(names, libraries, specializations, index=None):
    """Require a unique code owner for every profiler CUDA event, without prefixes."""
    index = origin_index(libraries) if index is None else index
    compiled = origin_index(specializations=specializations)
    records, unresolved = [], []
    for name in sorted(set(names)):
        normal = re.sub(r"\s+", "", name)
        candidates = {**index.get(normal, {}), **compiled.get(normal, {})}
        if len(candidates) != 1 or any(r["origin"] == "unaudited" for r in candidates.values()):
            unresolved.append({"name": name, "candidates": list(candidates.values())})
        else:
            owner = next(iter(candidates.values()))
            records.append({"profiler_name": name, **owner})
    return records, unresolved


class CodeAudit:
    def __init__(self, image_digest, out, *, inventory=IMAGE_CODE):
        self.image_digest, self.out = image_digest, Path(out)
        self.inventory_path = Path(inventory)
        self.loaded, self.images, self.records, self.saved = {}, {}, [], None
        self.case, self.unit = "preparation", None
        self.audit = None
        self.p2 = []

    def install(self, kernel_type=None):
        if kernel_type is None:
            from triton.compiler.compiler import CompiledKernel
            kernel_type = CompiledKernel
        original_init, original_run = kernel_type._init_handles, kernel_type.run
        if not isinstance(original_run, property):
            raise Unattested("unsupported pinned Triton run API")
        audit = self
        def initialize(kernel):
            if kernel.module is not None:
                return original_init(kernel)
            data = bytes(kernel.kernel)
            bound = definition(kernel)
            result = original_init(kernel)
            if kernel.kernel != data or definition(kernel) != bound:
                raise Invalid("Triton cubin changed during load")
            audit.loaded[id(kernel)] = (kernel, data, kernel.function, bound)
            return result
        def run(kernel):
            launch = original_run.__get__(kernel, kernel_type)
            def invoke(*args, **kwargs):
                if kwargs:
                    raise Unattested("unsupported Triton keyword launch parameters")
                loaded = audit.loaded.get(id(kernel))
                identity = specialization(kernel, args, loaded[1:] if loaded else None)
                if audit.unit is not None and not any(
                    all(identity[k] == r["identity"][k] for k in ("cubin_sha256", "entry", "target", "constants")) for r in audit.p2
                ):
                    raise Unattested("H1 P2 did not check this compiled Triton specialization")
                record = binding(audit.unit.get("unit_id") if audit.unit is not None else None, audit.case, identity)
                if record not in audit.records:
                    audit.records.append(record)
                if audit.unit is not None:
                    bindings = audit.unit.setdefault("triton_specializations", [])
                    if record not in bindings:
                        bindings.append(record)
                audit.out.mkdir(parents=True, exist_ok=True)
                path = audit.out / (identity["cubin_sha256"] + ".cubin")
                if path.exists() and sha256(path) != identity["cubin_sha256"]:
                    raise Invalid("bound Triton cubin snapshot changed")
                if not path.exists():
                    path.write_bytes(loaded[1])
                return launch(*args, **kwargs)
            return invoke
        self.saved = kernel_type, original_init, original_run
        kernel_type._init_handles, kernel_type.run = initialize, property(run)
        return self

    def close(self):
        if self.saved:
            kind, initialize, run = self.saved
            kind._init_handles, kind.run = initialize, run
            self.saved = None

    @contextmanager
    def context(self, case):
        previous = self.case
        self.case = case
        try:
            yield
        finally:
            self.case = previous

    def image_libraries(self):
        if not self.inventory_path.is_file():
            raise Unattested("pinned image library inventory is missing")
        inventory = json.loads(self.inventory_path.read_text())
        if inventory["image_digest"] != self.image_digest:
            raise Invalid("kernel inventory belongs to a different image")
        from tessera._dev.native_identity import loaded_native_identity
        paths = {line.split(None, 5)[5] for line in Path("/proc/self/maps").read_text().splitlines()
                 if len(line.split(None, 5)) == 6 and "x" in line.split()[1]}
        libraries = []
        for expected in inventory["libraries"]:
            path = expected["path"]
            if path not in paths:
                continue
            if str(Path(path).resolve()) != path:
                raise Invalid(f"image library is not at its image path: {path}")
            try:
                observed = loaded_native_identity(SimpleNamespace(__file__=path), self.inventory_path,
                                                   expected_sha256=expected["sha256"])
            except RuntimeError as error:
                raise Invalid(f"image library identity: {error}") from error
            if path not in self.images:
                self.images[path] = {"origin": "image", "path": path, "sha256": observed["sha256"],
                                     "symbols": image_symbols(path)}
            libraries.append(self.images[path])
        return libraries

    def unaudited_libraries(self, audited):
        """Account for every other executable mapped inode that can own CUDA code."""
        from experiments.numerics_rows.build import elf_sections
        from tessera._dev.native_identity import mapped_file_device
        import os

        known = {r["path"] for r in audited}
        libraries = []
        paths = {}
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(None, 5)
            if len(fields) == 6 and "x" in fields[1] and fields[5].startswith("/"):
                paths.setdefault(fields[5], []).append(fields)
        for path, mappings in paths.items():
            try:
                with open(path, "rb") as stream:
                    stat = os.fstat(stream.fileno())
                    device = mapped_file_device(stream.fileno())
                    if any((tuple(int(v, 16) for v in r[3].split(":")), int(r[4])) != (device, stat.st_ino) for r in mappings):
                        raise ValueError("mapped inode differs")
                    if path in known:
                        continue
                    data = stream.read()
                    _, _, sections = elf_sections(data)
                    if not any(r["name"] in (".nv_fatbin", ".nvFatBinSegment") for r in sections):
                        continue
                    verified = f"/proc/{os.getpid()}/fd/{stream.fileno()}"  # the checked inode, not a path that can change
                    symbols = image_symbols(verified)
                    record = {"origin": "unaudited", "path": path, "sha256": None, "symbols": symbols}
                    if not symbols:
                        record["empty_roster_proof"] = proved_empty_roster(path, data, stream.fileno())
                libraries.append(record)
            except (OSError, ValueError, RuntimeError) as error:
                cause = (str(error).splitlines() or [""])[0][:300]
                # Keep every other file's roster and proof: each failing file is its own UNATTESTED reason.
                libraries.append({"origin": "unaudited", "path": path, "sha256": None, "symbols": [], "error":
                                  f"{ROSTER_REFUSAL}{path}: {type(error).__name__}: {cause}"})
        return libraries

    def bind_profile(self, unit, names, native=()):
        # The full library audit runs again only when the mapped files change (path, inode, size or change time).
        # Native records are keyed by file, hash and parsed SASS roster, not by their long symbol lists.
        started = time.monotonic()
        key = json_digest([mapped_executables(), [[r["path"], r["sha256"], r.get("roster")] for r in native]])
        audited_at_case = self.audit is None or self.audit[0] != key
        if audited_at_case:
            libraries, refusal = list(native), None
            try:
                libraries.extend(self.image_libraries())
                libraries.extend(self.unaudited_libraries(libraries))
            except Unattested as error:
                refusal = str(error)
            self.audit = key, libraries, refusal, json_digest(libraries), origin_index(libraries)
        _, libraries, refusal, audit_sha256, index = self.audit
        audited = time.monotonic()
        if refusal:
            unit["unattested"].append(refusal)
        records, unresolved = kernel_origins(names, libraries, unit.get("triton_specializations", []), index)
        unit["kernel_origins"] = records
        unit["unresolved_kernel_origins"] = unresolved
        unit["dependency_code"] = {"observer": "IMG-N Triton 3.7.1 CompiledKernel load/run",
                                   "observation_complete": self.saved is not None,
                                   "p2_verified": all(any(all(r["identity"][k] == p["identity"][k] for k in ("cubin_sha256", "entry", "target", "constants")) for p in self.p2) for r in unit.get("triton_specializations", [])),
                                   "image_inventory_sha256": sha256(self.inventory_path) if self.inventory_path.is_file() else None,
                                   "proved_empty_rosters": [r["empty_roster_proof"] for r in libraries if "empty_roster_proof" in r],
                                   "library_audit": {"mapped_key_sha256": key, "audit_sha256": audit_sha256, "audited_at_case": audited_at_case},
                                   "seconds": {"audit": audited - started, "origins": time.monotonic() - audited}}
        if not unit["dependency_code"]["p2_verified"]:
            unit["unattested"].append("H1 P2 did not check a prepared Triton specialization")
        if unresolved:
            unit["unattested"].append("profiled kernel origin is missing or ambiguous")
        # Last, so that the case status (its last reason) names the roster cause, not only its effect.
        unit["unattested"].extend(r["error"] for r in libraries if "error" in r)


def check_unit_code(unit, image_digest=None, common=None, *, stream_units=()):
    if "code_stream" in unit:
        from experiments.numerics_rows.phase_artifact import read
        phase_path = Path(unit["code_stream"]["path"])
        header = json.loads(phase_path.read_text())
        found = {json_digest([u["module"], u["unit_id"]]): False for u in stream_units or [unit]}
        def check_case(case):
            key = json_digest([case["scope"]["module"], case["scope"]["unit_key"]])
            if key in found:
                proof = case.get("code")
                if case.get("completion") is not True or not proof or proof.get("unit_id") != case["scope"]["unit_key"] or "code_stream" in proof:
                    raise Invalid("unit case has no finished compiled-code proof")
                check_unit_code(proof, image_digest)
                found[key] = True
        document, digest = read(phase_path.parent, "compare", common if common is not None else header["identities"], header["scopes"], check_case=check_case)
        if digest != unit["code_stream"]["sha256"] or document["identities"].get("runtime", {}).get("image_digest") != image_digest:
            raise Invalid("unit code stream identity or digest differs")
        if not all(found.values()):
            raise Invalid("unit code stream has no matching cases")
        return
    evidence = unit.get("dependency_code", {})
    if evidence.get("observation_complete") is not True or not isinstance(unit.get("triton_specializations"), list):
        raise Invalid("unit has no finished compiled-code observation")
    names = {r["name"] for r in unit["arm"].get("profiler_kernels", []) + unit["arm"].get("preparation_kernels", [])}
    covered = [r["profiler_name"] for r in unit.get("kernel_origins", [])]
    if len(covered) != len(set(covered)) or names != set(covered) or unit.get("unresolved_kernel_origins"):
        raise Invalid("unit profiler kernels do not have complete unique code origins")
    specializations = unit["triton_specializations"]
    if specializations and evidence.get("p2_verified") is not True:
        raise Invalid("compiled specialization lacks its H1 P2 check")
    if len({json_digest(r) for r in specializations}) != len(specializations):
        raise Invalid("duplicate Triton specialization bindings")
    for record in specializations:
        identity = record["identity"]
        if record.get("sha256") != json_digest({k: v for k, v in record.items() if k != "sha256"}):
            raise Invalid("compiled specialization binding digest differs")
        if record.get("unit_id") != unit["unit_id"] or not isinstance(record.get("case"), str) or not record["case"] or set(identity) != {"cubin_sha256", "entry", "target", "constants", "launch"}:
            raise Invalid("incomplete Triton specialization binding")
        if not re.fullmatch("[0-9a-f]{64}", identity["cubin_sha256"]) or not identity["entry"] or not isinstance(identity["target"], dict) or identity["target"].get("backend") != "cuda":
            raise Invalid("malformed Triton specialization identity")
    for origin in unit.get("kernel_origins", []):
        if origin["origin"] not in ("tessera", "image", "compiled"):
            raise Invalid("unknown device kernel origin")
        if origin["origin"] == "compiled":
            if not any({k: v for k, v in r["identity"].items() if k != "launch"} == origin.get("code") for r in specializations):
                raise Invalid("kernel has no bound compiled identity")
        elif origin["origin"] == "image":
            inventory = json.loads(IMAGE_CODE.read_text())
            if inventory["image_digest"] != image_digest or not any(
                (r["path"], r["sha256"]) == (origin.get("path"), origin.get("sha256")) for r in inventory["libraries"]
            ):
                raise Invalid("kernel image-library origin differs from the pinned inventory")
        elif not any((r.get("path"), r.get("library_so_sha256")) == (origin.get("path"), origin.get("sha256"))
                     for r in unit["arm"].get("loaded_modules", [])):
            raise Invalid("kernel origin is not the audited Tessera library")
