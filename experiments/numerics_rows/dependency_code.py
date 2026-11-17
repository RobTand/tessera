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


def check_specializations(observed, expected):
    """A serve checks the complete unit/case/configuration binding before launch."""
    if not isinstance(expected, list):
        raise Unattested("Triton specialization bindings are missing")
    if observed not in expected:
        raise Invalid("Triton specialization identity differs from its unit and case")


def binding(unit_id, case, identity):
    record = {"unit_id": unit_id, "case": case, "identity": identity}
    return {**record, "sha256": json_digest(record)}


def symbol_aliases(functions):
    names = subprocess.check_output(["c++filt"], input="\n".join(functions) + "\n", text=True).splitlines() if functions else []
    return [{"symbol": symbol, "demangled": name} for symbol, name in zip(functions, names, strict=True)]


def image_symbols(path):
    """cuobjdump's ELF function roster avoids disassembling whole torch libraries."""
    result = subprocess.run(["cuobjdump", "-symbols", str(path)], capture_output=True, text=True)
    if result.returncode:
        raise Unattested(f"image CUDA symbol roster unavailable: {path}: {result.stderr}")
    # ELF entries printed by cuobjdump: STT_FUNC ... <exact mangled name>.
    functions = sorted({line.split()[-1] for line in result.stdout.splitlines() if re.search(r"\bSTT_FUNC\b", line)})
    return symbol_aliases(functions)


def kernel_origins(names, libraries, specializations):
    """Require a unique code owner for every profiler CUDA event, without prefixes."""
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
    records, unresolved = [], []
    for name in sorted(set(names)):
        candidates = index.get(normalize(name), {})
        if len(candidates) != 1:
            unresolved.append({"name": name, "candidates": list(candidates.values())})
        else:
            owner = next(iter(candidates.values()))
            records.append({"profiler_name": name, **owner})
    return records, unresolved


class CodeAudit:
    def __init__(self, image_digest, out, *, inventory=IMAGE_CODE, expected=None):
        self.image_digest, self.out = image_digest, Path(out)
        self.inventory_path = Path(inventory)
        self.loaded, self.images, self.records, self.saved = {}, {}, [], None
        self.case, self.unit = "preparation", None
        self.p2 = []
        self.expected = expected

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
                if audit.expected is not None:
                    check_specializations(record, audit.expected)
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

    def bind_profile(self, unit, names, native=()):
        libraries = list(native)
        try:
            libraries.extend(self.image_libraries())
        except Unattested as error:
            unit["unattested"].append(str(error))
        records, unresolved = kernel_origins(names, libraries, unit.get("triton_specializations", []))
        unit["kernel_origins"] = records
        unit["unresolved_kernel_origins"] = unresolved
        unit["dependency_code"] = {"observer": "IMG-N Triton 3.7.1 CompiledKernel load/run",
                                   "observation_complete": self.saved is not None,
                                   "p2_verified": all(any(all(r["identity"][k] == p["identity"][k] for k in ("cubin_sha256", "entry", "target", "constants")) for p in self.p2) for r in unit.get("triton_specializations", [])),
                                   "image_inventory_sha256": sha256(self.inventory_path) if self.inventory_path.is_file() else None}
        if not unit["dependency_code"]["p2_verified"]:
            unit["unattested"].append("H1 P2 did not check a prepared Triton specialization")
        if unresolved:
            unit["unattested"].append("profiled kernel origin is missing or ambiguous")


def check_unit_code(unit, image_digest=None):
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
